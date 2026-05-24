package com.altro.common.log.otlp

import com.altro.common.log.config.LogProperties
import org.apache.logging.log4j.core.LogEvent
import org.apache.logging.log4j.core.appender.AbstractAppender
import org.apache.logging.log4j.core.config.Property
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
    name: String,
    props: LogProperties,
    private val encoder: OtlpLogEncoder,
) : AbstractAppender(name, null, null, true, Property.EMPTY_ARRAY) {

    private val otlpProps = props.export

    private val httpClient = HttpClient.newBuilder()
        .connectTimeout(otlpProps.connectionTimeout)
        .build()

    private val queue = ArrayBlockingQueue<LogEvent>(otlpProps.queueCapacity)

    @Volatile
    private var workerThread: Thread? = null

    @Volatile
    private var running = false

    override fun append(event: LogEvent) {
        if (!running) {
            return
        }

        // Пытаемся положить в очередь. Если она полна — метод вернет false.
        val accepted = queue.offer(event.toImmutable())
        if (!accepted) {
            // Опционально: здесь можно инкрементировать метрику отброшенных логов
        }
    }

    override fun start() {
        // Проверяем статус через встроенные механизмы AbstractAppender
        if (isStarted) return

        super.start()

        running = true

        workerThread = Thread({ runWorkerLoop() }, "otlp-log-worker").apply {
            isDaemon = true
            start()
        }
    }

    private fun runWorkerLoop() {
        val batch = ArrayList<LogEvent>(otlpProps.batchSize)
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
                LOGGER.error("Error in OTLP worker loop", e)
            }
        }
    }

    private fun flush(eventsToSend: List<LogEvent>) {
        try {
            var payload = encoder.encodeBatch(eventsToSend)

            val shouldCompress = otlpProps.useGzip && payload.size > otlpProps.compressionThreshold.toBytes()
            if (shouldCompress) {
                payload = compressGzip(payload)
            }

            sendRequest(payload, shouldCompress)
        } catch (e: Exception) {
            LOGGER.error("Critical error during OTLP batch encoding", e)
        }
    }

    private fun compressGzip(data: ByteArray): ByteArray {
        val bos = ByteArrayOutputStream(data.size / GZIP_ESTIMATED_RATIO)
        GZIPOutputStream(bos).use { it.write(data) }
        return bos.toByteArray()
    }

    private fun sendRequest(payload: ByteArray, isCompressed: Boolean) {
        val rq = HttpRequest.newBuilder()
            .uri(otlpProps.url)
            .header(CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(otlpProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))
            .apply {
                if (isCompressed) {
                    header(CONTENT_ENCODING, ENCODING_GZIP)
                }
            }
            .build()

        try {
            val rs = httpClient.send(rq, HttpResponse.BodyHandlers.ofString())
            if (rs.statusCode() !in 200..299) {
                LOGGER.error("OTLP server returned error code: ${rs.statusCode()}, body: ${rs.body()}")
            }
        } catch (ex: Exception) {
            LOGGER.error("Network error while sending logs to OTLP", ex)
        }
    }

    override fun stop(timeout: Long, unit: TimeUnit): Boolean {
        running = false
        // Прерываем poll() воркера, чтобы он начал экстренно выгребать остатки логов [5.3]
        workerThread?.interrupt()

        val systemMs = unit.toMillis(timeout)
        val maxWaitMs = otlpProps.stopAwaitTimeout.toMillis()
        val waitTime = if (systemMs > 0) minOf(systemMs, maxWaitMs) else maxWaitMs

        try {
            workerThread?.join(waitTime)

            if (workerThread?.isAlive == true) {
                LOGGER.warn("OTLP worker termination timeout, some logs might be lost")
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        return super.stop(timeout, unit)
    }

    companion object {
        private const val ENCODING_GZIP = "gzip"
        private const val GZIP_ESTIMATED_RATIO = 2
    }
}
