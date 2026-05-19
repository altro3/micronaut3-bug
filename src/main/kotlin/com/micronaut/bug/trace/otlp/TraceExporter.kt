package com.micronaut.bug.trace.otlp

import com.micronaut.bug.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.CONTENT_ENCODING
import org.springframework.http.MediaType
import java.io.ByteArrayOutputStream
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.zip.GZIPOutputStream
import kotlin.time.Duration.Companion.milliseconds

class TraceExporter(
    appName: String,
    nodeName: String,
    traceProps: TraceProperties,
    private val httpClient: HttpClient,
) {
    private val log = KotlinLogging.logger {}

    private val exporterProps = traceProps.exporter
    private val encoder = OtlpTraceEncoder(appName, nodeName, traceProps)
    private val channel = Channel<TraceEvent>(exporterProps.queueCapacity)
    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var gzipByteArrayOutputStream = ByteArrayOutputStream(BASELINE_GZIP_BUFFER_SIZE)

    private val eventPool = ConcurrentLinkedQueue<TraceEvent>()

    fun enqueue(
        traceIdHex: String,
        spanIdHex: String,
        parentIdHex: String?,
        name: String,
        startEpochNanos: Long,
        endEpochNanos: Long,
        status: StatusCode,
        kind: Span.SpanKind,
        userAttrs: Map<String, Any>?,
        baggage: Map<String, String>?,
        propagationHeaders: Map<String, String>?,
        error: Throwable? = null,
    ) {
        if (!exporterProps.enabled) return

        // 1. Берем готовый объект из пула. Если пул пустой — создаем (он быстро прогреется)
        val event = eventPool.poll() ?: TraceEvent()

        // 2. Накатываем новые данные поверх старых (метод внутри TraceEvent)
        event.update(
            traceIdHex, spanIdHex, parentIdHex, name, startEpochNanos,
            endEpochNanos, status, kind, userAttrs, baggage, propagationHeaders, error
        )

        val result = channel.trySend(event)
        if (result.isFailure) {
            log.warn { "Trace queue overflow, span dropped: traceId=$traceIdHex" }

            // Если очередь переполнена, зануляем ссылки, чтобы не держать память, и возвращаем в пул
            event.clearReferences()
            eventPool.offer(event)
        }
    }

    @PostConstruct
    fun start() {
        if (!exporterProps.enabled) {
            log.info { "Trace batch exporter is disabled by configuration" }
            return
        }

        exportScope.launch {
            log.info { "Starting Trace batch exporter..." }

            val batch = ArrayList<TraceEvent>(exporterProps.batchSize)
            val flushIntervalMs = exporterProps.flushInterval.toMillis()

            while (isActive) {
                try {
                    val result = channel.receiveCatching() // Блокирующее ожидание первого элемента
                    if (result.isClosed) {
                        log.info { "Worker: channel closed, flushing remaining spans from queue..." }
                        flushRemainingOnShutdown(batch)
                        break
                    }

                    val first = result.getOrThrow()
                    batch.add(first)

                    val startTime = System.currentTimeMillis()

                    while (batch.size < exporterProps.batchSize) {
                        val nextResult = channel.tryReceive()

                        if (nextResult.isSuccess) {
                            val event = nextResult.getOrNull()
                            if (event != null) {
                                batch.add(event)
                            }
                        } else {
                            // ИСПРАВЛЕНИЕ 2: Если канал закрылся во время накопления,
                            // не ждем окончания таймера, выходим мгновенно для быстрого shutdown
                            if (nextResult.isClosed) {
                                break
                            }

                            // Если просто нет элементов, проверяем лимит времени пачки
                            if (System.currentTimeMillis() - startTime >= flushIntervalMs) {
                                break
                            }

                            // ИСПРАВЛЕНИЕ 1: Заменяем yield() на честный микро-сон,
                            // чтобы разгрузить CPU и не выжигать ядро процессора
                            delay(1.milliseconds)
                        }
                    }

                    sendBatch(batch)
                } catch (_: ClosedReceiveChannelException) {
                    log.info { "Worker: channel closed, flushing remaining spans from queue..." }
                    flushRemainingOnShutdown(batch)
                    break
                } catch (_: CancellationException) {
                    log.info { "Export loop stopped gracefully" }
                    break
                } catch (e: Exception) {
                    log.error(e) { "Error in Trace export loop. Retrying in ${exporterProps.retryInterval}..." }
                    delay(exporterProps.retryInterval.toMillis().milliseconds)
                } finally {
                    if (batch.isNotEmpty()) {
                        val currentBatchSize = batch.size
                        for (i in 0 until currentBatchSize) {
                            batch[i].clearReferences()
                        }
                        eventPool.addAll(batch)
                        batch.clear()
                    }
                }
            }
        }
    }

    @PreDestroy
    fun stop() {
        if (!exporterProps.enabled) return

        log.info { "Shutting down Trace exporter: closing channel..." }
        channel.close()

        runBlocking {
            val job = exportScope.coroutineContext[Job]
            val finished = withTimeoutOrNull(exporterProps.shutdownTimeout.toMillis().milliseconds) {
                job?.children?.forEach { it.join() }
                true
            }

            if (finished == null) {
                log.warn { "Worker timeout during graceful shutdown. Forcing stop." }
            }
        }

        exportScope.cancel()
        log.info { "Trace exporter stopped." }
    }

    private fun compressGzip(data: ByteArray): ByteArray {
        gzipByteArrayOutputStream.reset()
        GZIPOutputStream(gzipByteArrayOutputStream).use { it.write(data) }
        val result = gzipByteArrayOutputStream.toByteArray()

        // ЗАЩИТА ОТ РАЗДУВАНИЯ:
        if (gzipByteArrayOutputStream.size() > MAX_GZIP_BUFFER_SIZE) {
            log.info { "Gzip buffer resized from ${gzipByteArrayOutputStream.size()} bytes back to baseline due to peak burst" }
            gzipByteArrayOutputStream = ByteArrayOutputStream(BASELINE_GZIP_BUFFER_SIZE)
        }
        return result
    }

    private fun sendBatch(events: List<TraceEvent>) {
        if (events.isEmpty()) return

        try {
            var payload = encoder.encodeBatch(events)

            val shouldCompress = exporterProps.useGzip && payload.size > exporterProps.compressionThreshold.toBytes()
            if (shouldCompress) {
                payload = compressGzip(payload)
            }

            var attempts = 0
            var success = false
            while (attempts < 3 && !success) {
                attempts++
                success = sendRequest(payload, shouldCompress)
                if (!success && attempts < 3) {
                    Thread.sleep(exporterProps.retryInterval.toMillis())
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Critical error during Trace batch encoding" }
        }
    }

    private fun sendRequest(payload: ByteArray, isCompressed: Boolean): Boolean {
        val builder = HttpRequest.newBuilder()
            .uri(exporterProps.url)
            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(exporterProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))

        if (isCompressed) {
            builder.header(CONTENT_ENCODING, ENCODING_GZIP)
        }

        val rq = builder.build()

        return try {
            val rs = httpClient.send(rq, HttpResponse.BodyHandlers.discarding())
            if (rs.statusCode() in 200..299) {
                true
            } else {
                log.warn { "Trace rejected batch: code=${rs.statusCode()}" }
                false
            }
        } catch (ex: Exception) {
            log.warn { "Trace batch export failed due to network error: ${ex.message}" }
            false
        }
    }

    private fun flushRemainingOnShutdown(batch: ArrayList<TraceEvent>) {
        // ИСПРАВЛЕНИЕ БАГА: Строка batch.clear() УДАЛЕНА.
        // Мы сохраняем те элементы, которые воркер успел накопить перед закрытием канала.

        var result = channel.tryReceive()

        while (result.isSuccess) {
            val next = result.getOrNull()
            if (next != null) {
                batch.add(next)

                if (batch.size >= exporterProps.batchSize) {
                    sendBatch(batch)

                    val size = batch.size
                    for (i in 0 until size) {
                        batch[i].clearReferences()
                    }
                    eventPool.addAll(batch)

                    batch.clear()
                }
            }
            result = channel.tryReceive()
        }

        // Досылаем абсолютно всё, что осталось (включая пред-накопленные элементы)
        if (batch.isNotEmpty()) {
            sendBatch(batch)
            val size = batch.size
            for (i in 0 until size) {
                batch[i].clearReferences()
            }
            eventPool.addAll(batch)
            batch.clear() // Хвост занулен, finally в start() отработает вхолостую безопасно
        }
    }

    companion object {
        private const val ENCODING_GZIP = "gzip"
        private const val BASELINE_GZIP_BUFFER_SIZE = 16384
        private const val MAX_GZIP_BUFFER_SIZE = 65536
    }
}
