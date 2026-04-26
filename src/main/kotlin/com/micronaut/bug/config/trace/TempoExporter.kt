package com.micronaut.bug.config.trace

import com.micronaut.bug.util.TraceIdGenerator.toByteString
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
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpStatus
import org.springframework.http.HttpStatusCode
import org.springframework.http.MediaType
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

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

    /**
     * Собирает OTLP запрос и отправляет его в Tempo асинхронно.
     * Не блокирует вызывающий поток. Ошибки отправки подавляются во избежание рекурсивного логирования.
     *
     * @param traceIdHex ID трейса в HEX формате (32 символа).
     * @param spanIdHex ID текущего спана в HEX формате (16 символов).
     * @param parentIdHex ID родительского спана (если есть).
     * @param name Название операции (например, HTTP метод и путь).
     * @param startEpochNanos Время начала операции в наносекундах Unix Epoch.
     * @param durationNanos Длительность операции в наносекундах.
     * @param statusCode HTTP-статус (для подсветки ошибок в интерфейсе Grafana).
     * @param attrs Набор метаданных (заголовки, URI и т.д.) без тел.
     */
    fun sendTrace(
        traceIdHex: String,
        spanIdHex: String,
        parentIdHex: String?,
        name: String,
        startEpochNanos: Long,
        durationNanos: Long,
        statusCode: HttpStatusCode?,
        attrs: Map<String, String>,
        kind: Span.SpanKind = Span.SpanKind.SPAN_KIND_SERVER,
    ) {
        try {
            val spanBuilder = Span.newBuilder()
                .setTraceId(toByteString(traceIdHex))
                .setSpanId(toByteString(spanIdHex))
                .setName(name)
                .setKind(kind)
                .setStartTimeUnixNano(startEpochNanos)
                .setEndTimeUnixNano(startEpochNanos + durationNanos)

            // Помечаем спан как Error в UI Grafana, если статус >= 400
            if (statusCode == null || statusCode.isError) {
                val codeValue = statusCode?.value()
                val reasonPhrase = (statusCode as? HttpStatus)?.reasonPhrase ?: "HTTP $codeValue"
                spanBuilder.status = Status.newBuilder()
                    .setCode(StatusCode.STATUS_CODE_ERROR)
                    .setMessage(reasonPhrase)
                    .build()
            } else {
                spanBuilder.status = Status.newBuilder()
                    .setCode(StatusCode.STATUS_CODE_OK)
                    .build()
            }

            // Добавляем атрибуты
            attrs.forEach { (k, v) ->
                spanBuilder.addAttributes(
                    KeyValue.newBuilder()
                        .setKey(k)
                        .setValue(AnyValue.newBuilder().setStringValue(v).build())
                        .build()
                )
            }

            if (!parentIdHex.isNullOrBlank()) {
                spanBuilder.parentSpanId = toByteString(parentIdHex)
            }

            val rq = ExportTraceServiceRequest.newBuilder()
                .addResourceSpans(
                    ResourceSpans.newBuilder()
                        .setResource(
                            Resource.newBuilder()
                                .addAttributes(
                                    KeyValue.newBuilder()
                                        .setKey(ATTR_SERVICE_NAME)
                                        .setValue(AnyValue.newBuilder().setStringValue(appName).build())
                                        .build()
                                )
                                .build()
                        )
                        .addScopeSpans(
                            ScopeSpans.newBuilder()
                                .addSpans(spanBuilder.build())
                                .build()
                        )
                        .build()
                )
                .build()

            val httpRequest = HttpRequest.newBuilder()
                .uri(properties.url)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
                .POST(HttpRequest.BodyPublishers.ofByteArray(rq.toByteArray()))
                .build()

            val response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString())

            if (response.statusCode() !in 200..299) {
                log.error { "Tempo returned error code: ${response.statusCode()} | body: ${response.body()}" }
            }
//            // Асинхронная отправка: результат игнорируется, ошибки не прерывают выполнение
//            httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.discarding())
//                .exceptionally { null }

        } catch (e: Exception) {
            log.error { "Tempo export failed for $appName [traceId=$traceIdHex]: ${e.message}" }
        }
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
