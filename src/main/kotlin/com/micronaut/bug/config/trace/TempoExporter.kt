package com.micronaut.bug.config.trace

import com.micronaut.bug.config.trace.TraceIdGenerator.toByteString
import com.micronaut.bug.config.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.resource.v1.Resource
import io.opentelemetry.proto.trace.v1.ResourceSpans
import io.opentelemetry.proto.trace.v1.ScopeSpans
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy
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
import kotlin.coroutines.cancellation.CancellationException
import kotlin.time.Duration.Companion.milliseconds

/**
 * Высокопроизводительный экспортер трейсов, отправляющий данные в Tempo
 * напрямую по протоколу OTLP Protobuf через HTTP.
 *
 * @property appName Имя текущего сервиса (из spring.application.name).
 * @property properties Настройки подключения (URL, таймауты).
 */
class TempoExporter(
    private val appName: String,
    private val properties: TraceProperties,
    private val httpClient: HttpClient,
) {

    private val log = KotlinLogging.logger {}

    // Канал для накопления спанов. Capacity ограничивает память при перегрузке.
    private val channel = Channel<Span>(10000)

    // Отдельный Scope для фоновой работы экспортера
    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /**
     * Статический ресурс сервиса. Собирается один раз.
     */
    private val serviceResource: Resource by lazy {
        Resource.newBuilder()
            .addAttributes(kv(ATTR_SERVICE_NAME, appName))
            .build()
    }

    @PostConstruct
    fun start() {
        exportScope.launch {
            log.info { "Starting Tempo batch exporter for $appName" }
            while (isActive) {
                try {
                    val batch = mutableListOf<Span>()

                    // Блокирующее ожидание первого элемента
                    val first = channel.receive()
                    batch.add(first)

                    // Накапливаем остальные в течение окна времени из пропертей
                    withTimeoutOrNull(properties.flushInterval.toMillis().milliseconds) {
                        while (batch.size < properties.batchSize) {
                            batch.add(channel.receive())
                        }
                    }

                    sendBatch(batch)
                } catch (e: Exception) {
                    // Корректно завершаем корутину, если пришел сигнал отмены или канал закрыт
                    if (e is CancellationException || e is ClosedReceiveChannelException) {
                        log.info { "Export loop stopped gracefully" }
                        throw e
                    }
                    log.error(e) { "Error in Tempo export loop. Retrying in ${properties.retryInterval}..." }
                    delay(properties.retryInterval.toMillis().milliseconds) // Пауза при ошибке, чтобы не спамить в цикле
                }
            }
        }
    }

    /**
     * Помещает спан в очередь на отправку. Вызывается из NanoTracer.
     */
    fun enqueue(
        traceIdHex: String,
        spanIdHex: String,
        parentIdHex: String?,
        name: String,
        startEpochNanos: Long,
        endEpochNanos: Long,
        status: StatusCode = StatusCode.STATUS_CODE_UNSET,
        kind: Span.SpanKind = Span.SpanKind.SPAN_KIND_UNSPECIFIED,
        attrs: Map<String, Any?> = emptyMap(),
    ) {
        val spanBuilder = Span.newBuilder()
            .setTraceId(toByteString(traceIdHex))
            .setSpanId(toByteString(spanIdHex))
            .setName(name)
            .setKind(kind)
            .setStartTimeUnixNano(startEpochNanos)
            .setEndTimeUnixNano(endEpochNanos)
            .setStatus(Status.newBuilder().setCode(status))

        // Атрибуты
        attrs.forEach { (k, v) ->
            if (v != null) {
                spanBuilder.addAttributes(kv(k, v))
            }
        }

        if (!parentIdHex.isNullOrBlank()) {
            spanBuilder.parentSpanId = toByteString(parentIdHex)
        }

        // Отправляем в канал. Если канал переполнен, спан отбрасывается (защита памяти).
        val result = channel.trySend(spanBuilder.build())
        if (result.isFailure) {
            log.warn { "Trace queue overflow, span dropped: traceId=$traceIdHex" }
        }
    }

    private fun sendBatch(spans: List<Span>, async: Boolean = true) {
        val rq = ExportTraceServiceRequest.newBuilder()
            .addResourceSpans(
                ResourceSpans.newBuilder()
                    .setResource(serviceResource)
                    .addScopeSpans(
                        ScopeSpans.newBuilder()
                            .addAllSpans(spans)
                            .build()
                    )
                    .build()
            )
            .build()

        val httpRequest = HttpRequest.newBuilder()
            .uri(properties.url)
            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(properties.connectTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(rq.toByteArray()))
            .build()

        if (async) {
            httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.discarding())
                .whenComplete { rs, ex -> handleResponse(rs, ex) }
        } else {
            try {
                val rs = httpClient.send(httpRequest, HttpResponse.BodyHandlers.discarding())
                handleResponse(rs, null)
            } catch (ex: Exception) {
                handleResponse(null, ex)
            }
        }
    }

    private fun handleResponse(rs: HttpResponse<*>?, ex: Throwable?) {
        if (ex != null) {
            log.warn { "Tempo batch export failed: ${ex.message}" }
        } else if (rs?.statusCode() !in 200..299) {
            log.warn { "Tempo rejected batch: code=${rs?.statusCode()}" }
        }
    }

    @PreDestroy
    fun stop() {
        log.info { "Shutting down Tempo exporter: closing channel..." }

        // 1. Запрещаем новые поступления в канал
        channel.close()

        // 2. Даем небольшое время фоновой корутине на обработку (например, 5 секунд)
        runBlocking {
            val job = exportScope.coroutineContext[Job]
            // Ожидаем завершения цикла вычитки
            withTimeoutOrNull(5000.milliseconds) {
                // Пытаемся вычитать всё через flushRemaining вручную для надежности
                flushRemaining()
                job?.children?.forEach { it.join() }
            }
        }

        // 3. Окончательно отменяем всё
        exportScope.cancel()
        log.info { "Tempo exporter stopped." }
    }

    /**
     * Вычитывает все оставшиеся спаны из канала и отправляет их одним (или несколькими) батчами.
     */
    private fun flushRemaining() {
        val remaining = mutableListOf<Span>()
        // tryReceive выгребает всё, что есть в буфере в данный момент
        while (true) {
            val s = channel.tryReceive().getOrNull() ?: break
            remaining.add(s)
            if (remaining.size >= properties.batchSize) {
                sendBatch(ArrayList(remaining), async = false)
                remaining.clear()
            }
        }
        if (remaining.isNotEmpty()) {
            sendBatch(remaining, async = false)
        }
    }

    private fun kv(k: String, v: Any): KeyValue {
        val builder = KeyValue.newBuilder().setKey(k)
        val valueBuilder = AnyValue.newBuilder()

        when (v) {
            is String -> valueBuilder.setStringValue(v)
            is Long -> valueBuilder.setIntValue(v)
            is Int -> valueBuilder.setIntValue(v.toLong())
            is Boolean -> valueBuilder.setBoolValue(v)
            is Double -> valueBuilder.setDoubleValue(v)
            is Float -> valueBuilder.setDoubleValue(v.toDouble())
            else -> valueBuilder.setStringValue(v.toString())
        }

        return builder.setValue(valueBuilder.build()).build()
    }

    companion object {

        /**
         * Стандартные ключи атрибутов OpenTelemetry
         */
        const val ATTR_SERVICE_NAME = "service.name"
        const val ATTR_HTTP_METHOD = "http.method"
        const val ATTR_HTTP_URL = "http.url"
        const val ATTR_HTTP_STATUS_CODE = "http.status_code"

        /**
         * Кастомные ключи для логгинг-фильтров
         */
        const val ATTR_RQ_HEADERS = "rq.headers"
        const val ATTR_RS_HEADERS = "rs.headers"
        const val ATTR_CLIENT_ID = "client.id"
    }
}
