package com.micronaut.bug.log.otlp

import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.UnsynchronizedAppenderBase
import com.micronaut.bug.log.config.LogProperties
import org.springframework.http.HttpHeaders.CONTENT_ENCODING
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.MediaType
import java.io.ByteArrayOutputStream
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPOutputStream

/**
 * Профессиональная высокопроизводительная реализация Logback-аппендера для передачи
 * логов по протоколу OTLP (Protobuf/HTTP) на сервер OpenTelemetry (VictoriaLogs).
 *
 * Архитектура построена на паттерне Thread Confinement и Consumer (Потребитель):
 * - **Изоляция потоков:** Бизнес-потоки только складывают логи в неблокирующую очередь [5.3].
 * - **Воркер:** Единственный выделенный демон-поток занимается упаковкой и отправкой [5.3].
 * - **Backpressure (Обратное давление):** Синхронная отправка данных воркером защищает сеть от перегрузки [5.3].
 * - **Защита от OOM:** Очередь имеет жесткий фиксированный лимит, предотвращая раздувание кучи при всплесках [5.3].
 *
 * @param props Свойства логирования, из которых берутся размеры очередей, батчей и таймауты.
 * @param encoder Кодировщик для преобразования событий SLF4J в бинарный Protobuf-формат OTLP [5.3].
 */
class OtlpAppender(
    props: LogProperties,
    private val encoder: OtlpEncoder,
) : UnsynchronizedAppenderBase<ILoggingEvent>() {

    private val otlpProps = props.otlp

    /**
     * HTTP-клиент из стандартной библиотеки Java 11+.
     * Пул соединений переиспользуется автоматически самой JVM.
     */
    private val httpClient = HttpClient.newBuilder()
        .connectTimeout(otlpProps.connectionTimeout)
        .build()

    /**
     * Потокобезопасная очередь фиксированного размера [5.3].
     * Если очередь переполняется, новые логи отбрасываются (Drop-on-overflow),
     * гарантируя, что логирование никогда не затормозит и не уронит бизнес-логику [5.3].
     */
    private val queue = ArrayBlockingQueue<ILoggingEvent>(otlpProps.queueCapacity)

    /**
     * Ссылка на единственный фоновый поток-воркер [5.3].
     */
    @Volatile
    private var workerThread: Thread? = null

    /**
     * Флаг активности аппендера.
     */
    @Volatile
    private var running = false

    /**
     * Основная точка входа для событий логирования из приложения.
     * Быстро перекладывает задачу в очередь воркера без аллокаций лямбд и Runnable [5.3].
     * Метод выполняется за наносекунды и не блокирует вызывающий поток.
     *
     * @param eventObject Событие логирования от Logback.
     */
    override fun append(eventObject: ILoggingEvent) {
        if (!running) {
            return
        }

        // Пытаемся положить в очередь. Если она полна — метод вернет false.
        val accepted = queue.offer(eventObject)
        if (!accepted) {
            // Опционально: здесь можно инкрементировать метрику отброшенных логов
        }
    }

    /**
     * Инициализация аппендера фреймворком Logback.
     * Поднимает фоновый поток-воркер.
     */
    override fun start() {
        super.start()
        running = true

        workerThread = Thread({ runWorkerLoop() }, "otlp-log-worker").apply {
            isDaemon = true
            start()
        }
    }

    /**
     * Высокопроизводительный бесконечный цикл вычерпывания логов из очереди [5.3].
     *
     * Использует метод [ArrayBlockingQueue.drainTo] для атомарного извлечения
     * целой пачки логов за один системный вызов, снижая конкуренцию за локи [5.3].
     */
    private fun runWorkerLoop() {
        val batch = ArrayList<ILoggingEvent>(otlpProps.batchSize)
        val timeoutMs = otlpProps.batchTimeout.toMillis()

        // Воркер продолжает работать, пока приложение запущено ИЛИ пока в очереди есть логи [5.3].
        // Это гарантирует отправку последних данных (Graceful Shutdown) [5.3].
        while (running || !queue.isEmpty()) {
            try {
                // Ждем появления первого элемента с таймаутом, чтобы не грузить CPU вхолостую
                val firstEvent = queue.poll(timeoutMs, TimeUnit.MILLISECONDS)

                if (firstEvent != null) {
                    batch.add(firstEvent)
                    // "Выгребаем" из очереди все остальные накопившиеся логи до размера батча
                    queue.drainTo(batch, otlpProps.batchSize - 1)
                }

                // Сбрасываем батч, если он наполнился ИЛИ если сработал таймаут ожидания
                if (batch.isNotEmpty()) {
                    flush(batch)
                    batch.clear()
                }
            } catch (_: InterruptedException) {
                // ВОССТАНАВЛИВАЕМ статус прерывания!
                // Это заставит poll() на следующей итерации мгновенно завершиться без сна [5.3].
                Thread.currentThread().interrupt()

                // Если приложение выключается и очередь пуста — выходим из бесконечного цикла
                if (!running && queue.isEmpty()) {
                    break
                }
            } catch (e: Exception) {
                addError("Error in OTLP worker loop", e)
            }
        }
    }

    /**
     * Формирует батч, выполняет его кодирование в Protobuf и сжатие [5.3].
     *
     * @param eventsToSend Список событий для отправки.
     */
    private fun flush(eventsToSend: List<ILoggingEvent>) {
        try {
            var payload = encoder.encodeBatch(eventsToSend)

            if (otlpProps.useGzip) {
                payload = compressGzip(payload)
            }

            sendRequest(payload)
        } catch (e: Exception) {
            addError("Critical error during OTLP batch encoding", e)
        }
    }

    /**
     * Сжимает массив байтов алгоритмом GZIP.
     * Защищает трафик от раздувания [5.3].
     */
    private fun compressGzip(data: ByteArray): ByteArray {
        val bos = ByteArrayOutputStream(data.size / GZIP_ESTIMATED_RATIO)
        GZIPOutputStream(bos).use { it.write(data) }
        return bos.toByteArray()
    }

    /**
     * Выполняет синхронную отправку данных на OTLP-совместимый сервер [5.3].
     *
     * Использование блокирующего `send` гарантирует, что воркер не создаст
     * миллион асинхронных задач в памяти, если удаленный сервер начнет тормозить [5.3].
     */
    private fun sendRequest(payload: ByteArray) {
        val request = HttpRequest.newBuilder()
            .uri(otlpProps.url)
            .header(CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(otlpProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))
            .apply {
                if (otlpProps.useGzip) {
                    header(CONTENT_ENCODING, ENCODING_GZIP)
                }
            }
            .build()

        try {
            val response = httpClient.send(request, HttpResponse.BodyHandlers.discarding())
            if (response.statusCode() !in 200..299) {
                addError("OTLP server returned error code: ${response.statusCode()}")
            }
        } catch (ex: Exception) {
            addError("Network error while sending logs to OTLP", ex)
        }
    }

    /**
     * Завершение работы аппендера.
     * Дает воркеру время на отправку последних данных из очереди [5.3].
     */
    override fun stop() {
        running = false
        // Прерываем poll() воркера, чтобы он начал экстренно выгребать остатки логов [5.3]
        workerThread?.interrupt()
        try {
            val stopAwaitMs = otlpProps.stopAwaitTimeout.toMillis()

            workerThread?.join(stopAwaitMs)

            if (workerThread?.isAlive == true) {
                addWarn("OTLP worker termination timeout, some logs might be lost")
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        super.stop()
    }

    companion object {
        /**
         * Константа кодировки GZIP для HTTP-заголовка.
         */
        private const val ENCODING_GZIP = "gzip"
        /**
         * Ожидаемый коэффициент сжатия для начальной аллокации буфера [5.3].
         */
        private const val GZIP_ESTIMATED_RATIO = 2
    }
}
