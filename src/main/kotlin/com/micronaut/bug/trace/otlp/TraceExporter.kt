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
import org.springframework.http.MediaType
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import kotlin.time.Duration.Companion.milliseconds

class TraceExporter(
    appName: String,
    nodeName: String,
    private val httpClient: HttpClient,
    traceProps: TraceProperties,
) {
    private val log = KotlinLogging.logger {}
    private val exporterProps = traceProps.exporter

    private val encoder = OtlpTraceEncoder(appName, nodeName)

    private val channel = Channel<TraceEvent>(exporterProps.queueCapacity)

    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

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
        if (!exporterProps.enabled) {
            return
        }

        val event = TraceEvent(
            traceIdHex = traceIdHex,
            spanIdHex = spanIdHex,
            parentIdHex = parentIdHex,
            name = name,
            startEpochNanos = startEpochNanos,
            endEpochNanos = endEpochNanos,
            status = status,
            kind = kind,
            userAttrs = userAttrs,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            error = error,
        )

        val result = channel.trySend(event)
        if (result.isFailure) {
            log.warn { "Trace queue overflow, span dropped: traceId=$traceIdHex" }
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

            // Пул батча выделяется ОДИН раз во имя избежания аллокаций в цикле [5.3]
            val batch = ArrayList<TraceEvent>(exporterProps.batchSize)
            val flushIntervalMs = exporterProps.flushInterval.toMillis().milliseconds

            while (isActive) {
                try {
                    batch.clear()

                    // Ожидаем первый элемент пачки (блокирующий вызов)
                    val first = channel.receive()
                    batch.add(first)

                    // Накапливаем остальные спаны в рамках лимита времени
                    withTimeoutOrNull(flushIntervalMs) {
                        while (batch.size < exporterProps.batchSize) {
                            val next = channel.tryReceive().getOrNull() ?: break
                            batch.add(next)
                        }
                    }

                    sendBatch(batch)
                } catch (_: ClosedReceiveChannelException) {
                    log.info { "Worker: channel closed, finishing loop" }
                    break
                } catch (_: CancellationException) {
                    log.info { "Export loop stopped gracefully" }
                    break
                } catch (e: Exception) {
                    log.error(e) { "Error in Trace export loop. Retrying in ${exporterProps.retryInterval}..." }
                    delay(exporterProps.retryInterval.toMillis().milliseconds)
                }
            }
        }
    }

    private fun sendBatch(events: List<TraceEvent>) {
        if (events.isEmpty()) return

        val payload = encoder.encodeBatch(events)

        val httpRequest = HttpRequest.newBuilder()
            .uri(exporterProps.url)
            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(exporterProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))
            .build()

        try {
            // Прямой синхронный вызов. Корутина на Dispatchers.IO уснет на время I/O
            val rs = httpClient.send(httpRequest, HttpResponse.BodyHandlers.discarding())
            handleResponse(rs, null)
        } catch (ex: Exception) {
            handleResponse(null, ex)
        }
    }

    private fun handleResponse(rs: HttpResponse<*>?, ex: Throwable?) {
        if (ex != null) {
            log.warn { "Trace batch export failed: ${ex.message}" }
        } else if (rs?.statusCode() !in 200..299) {
            log.warn { "Trace rejected batch: code=${rs?.statusCode()}" }
        }
    }

    private fun flushRemaining() {
        val batch = ArrayList<TraceEvent>(exporterProps.batchSize)

        while (true) {
            val event = channel.tryReceive().getOrNull() ?: break
            batch.add(event)

            if (batch.size >= exporterProps.batchSize) {
                sendBatch(batch)
                batch.clear()
            }
        }

        if (batch.isNotEmpty()) {
            sendBatch(batch)
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
                log.warn { "Worker timeout, flushing remaining spans manually" }
                flushRemaining()
            }
        }

        exportScope.cancel()
        log.info { "Trace exporter stopped." }
    }
}
