package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType
import com.micronaut.bug.config.trace.NanoTracer
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_EXT_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_CLIENT_ID
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_ERROR_MESSAGE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_ERROR_TYPE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_METHOD
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_STATUS_CODE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_URL
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_RQ_HEADERS
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_RS_HEADERS
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse

/**
 * Инфраструктурный интерцептор для [org.springframework.web.client.RestTemplate].
 *
 * Отвечает за:
 * 1. Проброс идентификаторов сквозной трассировки (TraceId, SpanId) в заголовки исходящего запроса.
 * 2. Создание дочернего спана типа [Span.SpanKind.SPAN_KIND_CLIENT] для визуализации сетевого вызова в Tempo.
 * 3. Сбор детальных метаданных запроса и ответа (заголовки, URI, статус-коды).
 *
 * @property tracer Экземпляр [NanoTracer] для управления контекстом трейса.
 * @property selfServiceName Имя текущего сервиса для идентификации отправителя в распределенной системе.
 */
class NanoTraceClientInterceptor(
    private val tracer: NanoTracer,
    private val selfServiceName: String,
    private val httpClientProps: HttpClientProperties,
) : ClientHttpRequestInterceptor {

    override fun intercept(rq: HttpRequest, body: ByteArray, execution: ClientHttpRequestExecution): ClientHttpResponse {

        val fullUri = getFullUri(rq)
        // 1. Стартуем дочерний спан. Имя в формате "CLIENT: METHOD /path" для наглядности в UI Grafana.
        val ctx = tracer.startSpan(name = "CLIENT: ${rq.method} ${rq.uri.path}")

        // 2. Инъекция контекста в заголовки.
        // Текущий spanId становится parentId для принимающей стороны, обеспечивая связность дерева.
        if (httpClientProps.type == ClientType.INTERNAL) {
            rq.headers.add(HEADER_X_RQ_ID, ctx.traceId)
            rq.headers.add(HEADER_EXT_RQ_ID, ctx.spanId)
            // Sender тоже важен внутри сети
            rq.headers.add(HEADER_X_SENDER, selfServiceName)
        }

        return try {
            val rs = execution.execute(rq, body)

            // Фиксация успешного (или логического ошибочного, например 4xx/5xx) результата
            tracer.stop(
                ctx = ctx,
                status = if (rs.statusCode.isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                attrs = mapOf(
                    ATTR_HTTP_METHOD to rq.method.name(),
                    ATTR_HTTP_URL to fullUri,
                    ATTR_RQ_HEADERS to rq.headers.toString(),
                    ATTR_RS_HEADERS to rs.headers.toString(),
                    ATTR_CLIENT_ID to selfServiceName,
                    ATTR_HTTP_STATUS_CODE to rs.statusCode.value(),
                ),
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
            )
            rs
        } catch (e: Exception) {
            // Фиксация сетевых ошибок (Connection Refused, Timeout и т.д.)
            tracer.stop(
                ctx = ctx,
                status = Status.StatusCode.STATUS_CODE_ERROR,
                attrs = mapOf(
                    ATTR_HTTP_METHOD to rq.method.name(),
                    ATTR_HTTP_URL to fullUri,
                    ATTR_RQ_HEADERS to rq.headers.toString(),
                    ATTR_CLIENT_ID to selfServiceName,
                    ATTR_ERROR_MESSAGE to (e.message ?: e.javaClass.simpleName),
                    ATTR_ERROR_TYPE to e.javaClass.simpleName,
                ),
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
            )
            throw e
        }
    }

    /**
     * Формирует путь запроса с Query-параметрами для детального анализа в трейсинге.
     */
    private fun getFullUri(rq: HttpRequest): String {
        val uri = rq.uri
        return if (uri.query != null) "${uri.path}?${uri.query}" else uri.path
    }
}
