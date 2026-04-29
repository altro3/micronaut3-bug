package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.LoggingInterceptor.Companion.SLASH
import com.micronaut.bug.trace.NanoTraceFilter.Companion.METHODS_WITHOUT_BODY
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_TRACEPARENT
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_CLIENT
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SERVER
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_CLIENT
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_PEER_SERVICE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER_ADDRESS
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER_PORT
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_URL_FULL
import com.micronaut.bug.trace.TempoExporter.Companion.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_BAGGAGE
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_HTTP_RESPONSE_HEADER
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import org.slf4j.MDC
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

        val urlFull = if (rq.uri.isAbsolute) rq.uri.toString() else "$basePrefix${getFullUri(rq)}"

        // 1. Стартуем дочерний спан. Имя в формате "CLIENT: METHOD /path" для наглядности в UI Grafana.
        val ctx = tracer.startSpan(name = "${rq.method} $urlFull")

        tracer.getTraceParent()?.let {
            rq.headers.set(HEADER_TRACEPARENT, it)
        }

        // Проброс багажа (те самые заголовки из конфига)
        // Мы берем их из только что созданного активного контекста
        ctx.baggage.forEach { (name, value) ->
            if (!rq.headers.keys.any { it.equals(name, ignoreCase = true) }) {
                rq.headers.set(name, value)
            }
        }

        // Sender тоже важен внутри сети
        if (props.type == INTERNAL) {
            rq.headers.set(HEADER_X_SENDER, selfServiceName)
        }

        val originalClient = MDC.get(MDC_CLIENT)
        val originalServer = MDC.get(MDC_SERVER)

        return try {
            val rs = execution.execute(rq, body)
            val isError = rs.statusCode.isError

            MDC.put(MDC_CLIENT, selfServiceName)
            MDC.put(MDC_SERVER, props.serviceName)

            // Фиксация успешного (или логического ошибочного, например 4xx/5xx) результата
            tracer.stop(
                ctx = ctx,
                status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_OK,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                attrs = buildMap {
                    put(ATTR_HTTP_REQUEST_METHOD, rq.method.name())
                    put(ATTR_URL_FULL, urlFull)
                    put(ATTR_HTTP_RESPONSE_STATUS_CODE, rs.statusCode.value().toLong())

                    // 2. SERVER ADDRESS: Берем хост из URI запроса, если его нет — из пропертей
                    val host = rq.uri.host ?: props.url.host ?: HOST_UNKNOWN
                    put(ATTR_SERVER_ADDRESS, host)
                    val port = rq.uri.port.takeIf { it != -1 }
                        ?: props.url.port.takeIf { it != -1 }
                        ?: if (urlFull.startsWith(PROTOCOL_HTTPS)) 443 else 80
                    put(ATTR_SERVER_PORT, port.toLong())

                    put(ATTR_CLIENT, selfServiceName) // Имя нашего сервиса (например, "service1")
                    put(ATTR_SERVER, props.serviceName.toString()) // Имя цели (например, "service2")

                    props.serviceName?.let { put(ATTR_PEER_SERVICE, it) }

                    // Размеры
                    if (rq.method.name() !in METHODS_WITHOUT_BODY) {
                        val reqSize = body.size.toLong()
                        if (reqSize > 0) {
                            put(ATTR_HTTP_REQUEST_BODY_SIZE, reqSize)
                        }
                    }

                    rs.headers.contentLength.takeIf { it != -1L }?.let {
                        put(ATTR_HTTP_RESPONSE_BODY_SIZE, it)
                    }

                    if (isError) {
                        put(ATTR_EXCEPTION_MESSAGE, "HTTP ${rs.statusCode.value()}")
                    }

                    ctx.baggage.forEach { (name, value) ->
                        put("$PREFIX_BAGGAGE$name", value)
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
                    put(ATTR_CLIENT, selfServiceName) // Имя нашего сервиса (например, "service1")
                    put(ATTR_SERVER, props.serviceName.toString()) // Имя цели (например, "service2")
                    put(ATTR_EXCEPTION_TYPE, e.javaClass.name)
                    put(ATTR_EXCEPTION_MESSAGE, e.message ?: e.javaClass.simpleName)
                    put(ATTR_EXCEPTION_STACKTRACE, e.stackTraceToString())
                    putAll(getRequestHeadersAttrs(rq))
                }
            )
            throw e
        } finally {
            MDC.put(MDC_CLIENT, originalClient)
            MDC.put(MDC_SERVER, originalServer)
        }
    }

    private fun getRequestHeadersAttrs(rq: HttpRequest): Map<String, List<String>> =
        rq.headers.keys
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
                val key = "${PREFIX_HTTP_REQUEST_HEADER}${name.lowercase()}"
                key to (rq.headers[name] ?: emptyList())
            }

    private fun getResponseHeadersAttrs(rs: ClientHttpResponse): Map<String, List<String>> =
        rs.headers.keys
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
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

    companion object {

        const val PROTOCOL_HTTP = "http"
        const val PROTOCOL_HTTPS = "https"

        const val HOST_UNKNOWN = "unknown"
    }
}
