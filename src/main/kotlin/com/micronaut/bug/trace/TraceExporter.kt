package com.micronaut.bug.trace

import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_DEPLOYMENT_ENVIRONMENT
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVICE_NAME
import com.micronaut.bug.trace.TraceIdGenerator.toByteString
import com.micronaut.bug.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.ArrayValue
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
 * Высокопроизводительный пакетный экспортер трейсов для Grafana Tempo.
 *
 * Осуществляет отправку данных напрямую по протоколу OTLP (OpenTelemetry Line Protocol)
 * в формате Protobuf через HTTP. Для минимизации нагрузки на сеть и CPU спаны
 * накапливаются во внутренней очереди [Channel] и отправляются группами (батчами).
 *
 * @property appName Имя текущего сервиса для идентификации ресурса в системе трассировки.
 * @property httpClient Клиент для выполнения HTTP-запросов к API Tempo.
 * @param traceProps Глобальные настройки трейсинга, из которых извлекаются [ExporterProperties].
 */
class TraceExporter(
    private val appName: String,
    private val nodeName: String,
    private val httpClient: HttpClient,
    traceProps: TraceProperties,
) {

    private val log = KotlinLogging.logger {}

    private val exporterProps = traceProps.exporter

    // Канал для накопления спанов. Capacity ограничивает память при перегрузке.
    private val channel = Channel<Span>(exporterProps.queueCapacity)

    // Отдельный Scope для фоновой работы экспортера
    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /**
     * Статический ресурс сервиса. Собирается один раз.
     */
    private val serviceResource: Resource by lazy {
        Resource.newBuilder()
            .addAttributes(kv(ATTR_SERVICE_NAME, appName))
            .addAttributes(kv(ATTR_DEPLOYMENT_ENVIRONMENT, nodeName))
            .build()
    }

    @PostConstruct
    fun start() {

        // Если экспорт выключен, воркер даже не заведется
        if (!exporterProps.enabled) {
            log.info { "Tempo batch exporter is disabled by configuration" }
            return
        }

        exportScope.launch {
            log.info { "Starting Tempo batch exporter for $appName" }
            while (isActive) {
                try {
                    val batch = mutableListOf<Span>()

                    // Блокирующее ожидание первого элемента
                    val first = channel.receive()
                    batch.add(first)

                    // Накапливаем остальные в течение окна времени из пропертей
                    withTimeoutOrNull(exporterProps.flushInterval.toMillis().milliseconds) {
                        while (batch.size < exporterProps.batchSize) {
                            // Используем tryReceive вместо receive, чтобы таймаут
                            // не прервал корутину в момент извлечения данных
                            val next = channel.tryReceive().getOrNull() ?: break
                            batch.add(next)
                        }
                    }

                    sendBatch(batch)
                } catch (_: ClosedReceiveChannelException) {
                    // Канал закрыт, данных больше не будет.
                    // Просто выходим из цикла, чтобы корутина завершилась мирно.
                    log.info { "Worker: channel closed, finishing loop" }
                    break
                } catch (_: CancellationException) {
                    // Корректно завершаем корутину, если пришел сигнал отмены или канал закрыт
                    log.info { "Export loop stopped gracefully" }
                    break
                } catch (e: Exception) {
                    log.error(e) { "Error in Tempo export loop. Retrying in ${exporterProps.retryInterval}..." }
                    delay(exporterProps.retryInterval.toMillis().milliseconds) // Пауза при ошибке, чтобы не спамить в цикле
                }
            }
        }
    }

    @PreDestroy
    fun stop() {

        // Если экспорт не был включен, нам нечего останавливать и очищать
        if (!exporterProps.enabled) {
            return
        }

        log.info { "Shutting down Tempo exporter: closing channel..." }

        // 1. Запрещаем новые поступления в канал
        channel.close()

        // 2. Даем небольшое время фоновой корутине на обработку (например, 5 секунд)
        runBlocking {
            val job = exportScope.coroutineContext[Job]

            // 1. Сначала даем воркеру штатно дочитать канал (join)
            val finished = withTimeoutOrNull(exporterProps.shutdownTimeout.toMillis().milliseconds) {
                job?.children?.forEach { it.join() }
                true
            }

            // 2. И ТОЛЬКО ЕСЛИ он не успел по таймауту, вычищаем остатки сами
            if (finished == null) {
                log.warn { "Worker timeout, flushing remaining spans manually" }
                flushRemaining()
            }
        }

        // 3. Окончательно отменяем всё
        exportScope.cancel()
        log.info { "Tempo exporter stopped." }
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

        // Если экспорт выключен, выходим сразу, не собирая билдеры
        if (!exporterProps.enabled) {
            return
        }

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
            .uri(exporterProps.url)
            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(exporterProps.requestTimeout)
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

    /**
     * Очистка очереди перед завершением работы.
     * Выгребает все доступные спаны и отправляет их батчами.
     */
    private fun flushRemaining() {
        // tryReceive выгребает всё, что есть в буфере в данный момент
        val batch = mutableListOf<Span>()

        // Генерируем последовательность из элементов канала, пока они не кончатся
        generateSequence { channel.tryReceive().getOrNull() }
            .forEach { span ->
                batch.add(span)

                // Если набрали полный батч — отправляем и чистим список
                if (batch.size >= exporterProps.batchSize) {
                    sendBatch(batch, async = false)
                    batch.clear()
                }
            }

        // Отправляем остатки, если они есть
        if (batch.isNotEmpty()) {
            sendBatch(batch, async = false)
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
            is Iterable<*> -> {
                val arrayBuilder = ArrayValue.newBuilder()
                v.forEach { item ->
                    val itemValue = AnyValue.newBuilder().setStringValue(item.toString()).build()
                    arrayBuilder.addValues(itemValue)
                }
                valueBuilder.setArrayValue(arrayBuilder)
            }

            else -> valueBuilder.setStringValue(v.toString())
        }

        return builder.setValue(valueBuilder.build()).build()
    }
}
