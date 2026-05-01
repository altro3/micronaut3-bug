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

class OtlpAppender(
    props: LogProperties,
    private val encoder: OtlpEncoder,
) : UnsynchronizedAppenderBase<ILoggingEvent>() {

    private val otlpProps = props.otlp

    private val httpClient = HttpClient.newBuilder()
        .connectTimeout(otlpProps.connectionTimeout)
        .build()

    /**
     * ШАГ 1: Потокобезопасная очередь ФИКСИРОВАННОГО размера.
     * Защищает приложение от OOM при всплесках трафика.
     */
    private val queue = ArrayBlockingQueue<ILoggingEvent>(otlpProps.queueCapacity)

    /**
     * ШАГ 2: Единственный выделенный поток-воркер (Daemon).
     */
    @Volatile
    private var workerThread: Thread? = null

    @Volatile
    private var running = false

    /**
     * Основная точка входа. Бизнес-потоки только складывают логи в очередь.
     * Никаких аллокаций Runnable и лямбд. Метод выполняется за наносекунды.
     */
    override fun append(eventObject: ILoggingEvent) {
        if (!running) {
            return
        }

        // Пытаемся положить в очередь. Если она полна — лог отбрасывается.
        // Это гарантирует, что логирование никогда не затормозит бизнес-логику.
        val accepted = queue.offer(eventObject)
        if (!accepted) {
            // Опционально: инкремент метрики отброшенных логов
        }
    }

    override fun start() {
        super.start()
        running = true

        workerThread = Thread({ runWorkerLoop() }, "otlp-log-worker").apply {
            isDaemon = true
            start()
        }
    }

    /**
     * ШАГ 3: Высокопроизводительный бесконечный цикл вычерпывания логов.
     */
    private fun runWorkerLoop() {
        val batch = ArrayList<ILoggingEvent>(otlpProps.batchSize)
        val timeoutMs = otlpProps.batchTimeout.toMillis()

        while (running || !queue.isEmpty()) {
            try {
                val firstEvent = queue.poll(timeoutMs, TimeUnit.MILLISECONDS)

                if (firstEvent != null) {
                    batch.add(firstEvent)
                    queue.drainTo(batch, otlpProps.batchSize - 1)
                }

                if (batch.isNotEmpty()) {
                    flush(batch)
                    batch.clear()
                }
            } catch (_: InterruptedException) {
                // ВОССТАНАВЛИВАЕМ СТАТУС ПРЕРЫВАНИЯ!
                // Это заставит poll() на следующей итерации мгновенно завершиться без сна.
                Thread.currentThread().interrupt()

                // Если мы выключаемся и очередь пуста — только тогда выходим
                if (!running && queue.isEmpty()) {
                    break
                }
            } catch (e: Exception) {
                addError("Error in OTLP worker loop", e)
            }
        }
    }

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
     */
    private fun compressGzip(data: ByteArray): ByteArray {
        val bos = ByteArrayOutputStream(data.size / GZIP_ESTIMATED_RATIO)
        GZIPOutputStream(bos).use { it.write(data) }
        return bos.toByteArray()
    }

    /**
     * Выполняет асинхронную отправку данных на OTLP-совместимый сервер.
     */
    private fun sendRequest(payload: ByteArray) {
        val requestBuilder = HttpRequest.newBuilder()
            .uri(otlpProps.url)
            .header(CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(otlpProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))

        if (otlpProps.useGzip) {
            requestBuilder.header(CONTENT_ENCODING, ENCODING_GZIP)
        }

        try {
            val rs = httpClient.send(requestBuilder.build(), HttpResponse.BodyHandlers.discarding())
            if (rs.statusCode() !in 200..299) {
                addError("OTLP server returned error code: ${rs.statusCode()}")
            }
        } catch (ex: Exception) {
            addError("Network error while sending logs to OTLP", ex)
        }
    }

    /**
     * Завершение работы аппендера.
     * Дает воркеру 5 секунд на отправку последних данных.
     */
    override fun stop() {
        running = false
        // Прерываем poll() воркера, чтобы он начал экстренно выгребать остатки логов
        workerThread?.interrupt()
        try {
            // Используем динамический таймаут из конфигурации
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
         * Константа кодировки GZIP.
         */
        private const val ENCODING_GZIP = "gzip"

        /**
         * Ожидаемый коэффициент сжатия для аллокации буфера.
         */
        private const val GZIP_ESTIMATED_RATIO = 2
    }
}
