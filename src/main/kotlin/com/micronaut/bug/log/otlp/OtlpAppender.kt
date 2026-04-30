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
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPOutputStream

/**
 * Профессиональная реализация Logback-аппендера для передачи логов по протоколу OTLP (Protobuf/HTTP).
 *
 * Особенности:
 * - **Batching**: Накопление логов в пачки для снижения нагрузки на сеть.
 * - **Non-blocking**: Отдельный воркер-поток (Thread Confinement) для обработки и упаковки логов.
 * - **Compression**: Поддержка GZIP-сжатия бинарного Protobuf-пейлоада.
 * - **Spring Integration**: Использование стандартных констант Spring для заголовков и типов контента.
 */
class OtlpAppender(
    props: LogProperties,
    private val encoder: OtlpEncoder,
) : UnsynchronizedAppenderBase<ILoggingEvent>() {

    private val otlpProps = props.otlp

    private val httpClient = HttpClient.newBuilder()
        .connectTimeout(otlpProps.connectionTimeout)
        .build()

    /**
     * Список для накопления батча (доступен только воркеру).
     */
    private val batch = mutableListOf<ILoggingEvent>()

    /**
     * Воркер для изоляции тяжелых операций (сериализация, сжатие) от бизнес-потоков.
     */
    private val executor = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable, "otlp-log-worker").apply { isDaemon = true }
    }

    /**
     * Основная точка входа для событий логирования.
     * Быстро перекладывает задачу в очередь воркера.
     */
    override fun append(eventObject: ILoggingEvent) {
        executor.execute {
            batch.add(eventObject)
            if (batch.size >= otlpProps.batchSize) {
                flush()
            }
        }
    }

    /**
     * Инициализация аппендера.
     * Запускает периодический сброс накопленного батча по таймауту.
     */
    override fun start() {
        super.start()
        val timeoutMs = otlpProps.batchTimeout.toMillis()
        executor.scheduleWithFixedDelay({
            executor.execute { flush() }
        }, timeoutMs, timeoutMs, TimeUnit.MILLISECONDS)
    }

    /**
     * Формирует батч, выполняет его кодирование и сжатие.
     */
    private fun flush() {
        if (batch.isEmpty()) {
            return
        }

        val eventsToSend = ArrayList(batch)
        batch.clear()

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

        httpClient.sendAsync(requestBuilder.build(), HttpResponse.BodyHandlers.discarding())
            .thenAccept { response ->
                if (response.statusCode() !in 200..299) {
                    addError("OTLP server returned error code: ${response.statusCode()}")
                }
            }
            .exceptionally { ex ->
                addError("Network error while sending logs to OTLP", ex)
                null
            }
    }

    /**
     * Завершение работы аппендера.
     * Дает воркеру 5 секунд на отправку последних данных.
     */
    override fun stop() {
        executor.execute { flush() }
        executor.shutdown()
        try {
            if (!executor.awaitTermination(STOP_AWAIT_SECONDS, TimeUnit.SECONDS)) {
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

        /**
         * Время ожидания завершения работы воркера.
         */
        private const val STOP_AWAIT_SECONDS = 5L
    }
}
