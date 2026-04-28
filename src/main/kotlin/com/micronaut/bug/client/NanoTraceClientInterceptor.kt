package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType
import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.SLASH
import com.micronaut.bug.config.trace.NanoTracer
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_EXT_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_TRACEPARENT
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_SERVER_ADDRESS
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_SERVER_PORT
import com.micronaut.bug.config.trace.TempoExporter.Companion.ATTR_URL_FULL
import com.micronaut.bug.config.trace.TempoExporter.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.config.trace.TempoExporter.Companion.PREFIX_HTTP_RESPONSE_HEADER
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status.StatusCode
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
    private val props: HttpClientProperties,
) : ClientHttpRequestInterceptor {

    /**
     * Префикс для формирования полного URI. Вычисляется один раз при создании интерцептора.
     */
    private val basePrefix: String = props.url.toString().removeSuffix(SLASH)

    override fun intercept(rq: HttpRequest, body: ByteArray, execution: ClientHttpRequestExecution): ClientHttpResponse {

        val fullUri = getFullUri(rq)
        // 1. Стартуем дочерний спан. Имя в формате "CLIENT: METHOD /path" для наглядности в UI Grafana.
        val ctx = tracer.startSpan(name = "${rq.method} ${rq.uri.path}")

        tracer.getTraceParent()?.let {
            rq.headers.set(HEADER_TRACEPARENT, it)
        }

        // Legacy поддержка для внутренних сервисов
        if (props.type == ClientType.INTERNAL) {
            rq.headers.set(HEADER_X_RQ_ID, ctx.traceId)
            rq.headers.set(HEADER_EXT_RQ_ID, ctx.spanId)
            // Sender тоже важен внутри сети
            rq.headers.set(HEADER_X_SENDER, selfServiceName)
        }

        return try {
            val rs = execution.execute(rq, body)
            val isError = rs.statusCode.isError

            // Фиксация успешного (или логического ошибочного, например 4xx/5xx) результата
            tracer.stop(
                ctx = ctx,
                status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_OK,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                attrs = buildMap {
                    put(ATTR_HTTP_REQUEST_METHOD, rq.method.name())

                    val urlFull = if (rq.uri.isAbsolute) rq.uri.toString() else "$basePrefix${getFullUri(rq)}"
                    put(ATTR_URL_FULL, urlFull)

                    put(ATTR_HTTP_RESPONSE_STATUS_CODE, rs.statusCode.value().toLong())

                    // 2. SERVER ADDRESS: Берем хост из URI запроса, если его нет — из пропертей
                    val host = rq.uri.host ?: props.url.host ?: "unknown"
                    put(ATTR_SERVER_ADDRESS, host)
                    val port = rq.uri.port.takeIf { it != -1 }
                        ?: props.url.port.takeIf { it != -1 }
                        ?: if (urlFull.startsWith("https")) 443 else 80
                    put(ATTR_SERVER_PORT, port.toLong())

                    // Размеры
                    val reqSize = body.size.toLong()
                    if (reqSize > 0) {
                        put(ATTR_HTTP_REQUEST_BODY_SIZE, reqSize)
                    }

                    rs.headers.contentLength.takeIf { it != -1L }?.let {
                        put(ATTR_HTTP_RESPONSE_BODY_SIZE, it)
                    }

                    if (isError) {
                        put(ATTR_EXCEPTION_MESSAGE, "HTTP ${rs.statusCode.value()}")
                    }

                    // Заголовки (используем тот же подход string[])
                    putAll(getRequestHeadersAttrs(rq))
                    putAll(getResponseHeadersAttrs(rs))
                }
            )
            rs
        } catch (e: Exception) {
            // Фиксация сетевых ошибок (Connection Refused, Timeout и т.д.)
            tracer.stop(
                ctx = ctx,
                status = StatusCode.STATUS_CODE_ERROR,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                attrs = buildMap {
                    put(ATTR_HTTP_REQUEST_METHOD, rq.method.name())
                    put(ATTR_URL_FULL, rq.uri.toString())
                    put(ATTR_EXCEPTION_TYPE, e.javaClass.name)
                    put(ATTR_EXCEPTION_MESSAGE, e.message ?: e.javaClass.simpleName)
                    put(ATTR_EXCEPTION_STACKTRACE, e.stackTraceToString())
                    putAll(getRequestHeadersAttrs(rq))
                }
            )
            throw e
        }
    }

    private fun getRequestHeadersAttrs(rq: HttpRequest): Map<String, List<String>> =
        rq.headers.keys.associate { name ->
            val key = "${PREFIX_HTTP_REQUEST_HEADER}${name.lowercase()}"
            key to (rq.headers[name] ?: emptyList())
        }

    private fun getResponseHeadersAttrs(rs: ClientHttpResponse): Map<String, List<String>> =
        rs.headers.keys.associate { name ->
            val key = "${PREFIX_HTTP_RESPONSE_HEADER}${name.lowercase()}"
            key to (rs.headers[name] ?: emptyList())
        }

    /**
     * Формирует путь запроса с Query-параметрами для детального анализа в трейсинге.
     */
    private fun getFullUri(rq: HttpRequest): String {
        val uri = rq.uri
        return if (uri.query != null) "${uri.path}?${uri.query}" else uri.path
    }
}
