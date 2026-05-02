package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.LoggingInterceptor.Companion.SLASH
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_CLIENT
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_PEER_SERVICE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER_ADDRESS
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER_PORT
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_URL_FULL
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_TRACEPARENT
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SOURCE
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SPAN_ID
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_TARGET
import com.micronaut.bug.trace.NanoTracer.Companion.METHODS_WITHOUT_BODY
import com.micronaut.bug.trace.NanoTracer.Companion.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.formatBaggage
import io.github.oshai.kotlinlogging.KotlinLogging
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

    private val log = KotlinLogging.logger {}

    /**
     * Префикс для формирования полного URI. Вычисляется один раз при создании интерцептора.
     */
    private val basePrefix: String = props.url.toString().removeSuffix(SLASH)

    override fun intercept(rq: HttpRequest, body: ByteArray, execution: ClientHttpRequestExecution): ClientHttpResponse {
        val startTimeNano = System.nanoTime()
        val urlFull = if (rq.uri.isAbsolute) rq.uri.toString() else "$basePrefix${getFullUri(rq)}"

        // Стартуем дочерний спан. Имя в формате "CLIENT: METHOD /path" для наглядности в UI Grafana.
        val ctx = tracer.startSpan(name = "$PREFIX_CLIENT_SPAN ${rq.method} $urlFull")

        // Проброс стандартного контекста трейсинга (W3C traceparent)
        tracer.getTraceParent()?.let {
            rq.headers.set(HEADER_TRACEPARENT, it)
        }

        // Идентификация отправителя для внутренних вызовов
        if (props.type == INTERNAL) {
            rq.headers.set(HEADER_X_SENDER, selfServiceName)
        }

        // Проброс стандартного Baggage (W3C)
        formatBaggage(ctx.baggage)?.let {
            rq.headers.set(HEADER_BAGGAGE, it)
        }

        // Проброс Tracestate по стандарту W3C
        ctx.traceState?.let {
            if (!rq.headers.containsKey(NanoTracer.HEADER_TRACESTATE)) {
                rq.headers.set(NanoTracer.HEADER_TRACESTATE, it)
            }
        }

        // Проброс кастомных заголовков (Propagation Headers)
        // Мы берем их из контекста и проставляем "как есть"
        if (!ctx.propagationHeaders.isNullOrEmpty()) {
            val keys = rq.headers.keys
            for (entry in ctx.propagationHeaders.entries) {
                val name = entry.key
                // Быстрая проверка без создания лямбд
                var found = false
                for (k in keys) {
                    if (k.equals(name, ignoreCase = true)) {
                        found = true
                        break
                    }
                }
                if (!found) {
                    rq.headers.set(name, entry.value)
                }
            }
        }

        val originalClient = MDC.get(MDC_SOURCE)
        val originalServer = MDC.get(MDC_TARGET)

        return try {
            val rs = execution.execute(rq, body)

            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            // Берем порог из настроек конкретного клиента
            val isSlow = durationMs >= props.log.slowThreshold.toMillis()

            if (isSlow) {
                log.warn { "Slow client response: ${rq.method} $urlFull took ${durationMs}ms (threshold: ${props.log.slowThreshold.toMillis()}ms) [extRqId: ${MDC.get(MDC_SPAN_ID)}]" }
            }

            val isError = rs.statusCode.isError

            MDC.put(MDC_SOURCE, selfServiceName)
            MDC.put(MDC_TARGET, props.serviceName)

            // Собираем ТОЛЬКО кастомные атрибуты.
            val attrs = HashMap<String, Any>(32)
            attrs[ATTR_HTTP_REQUEST_METHOD] = rq.method.name()
            attrs[ATTR_URL_FULL] = urlFull
            attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = rs.statusCode.value().toLong()

            val host = rq.uri.host ?: props.url.host ?: HOST_UNKNOWN
            attrs[ATTR_SERVER_ADDRESS] = host
            val port = rq.uri.port.takeIf { it != -1 }
                ?: props.url.port.takeIf { it != -1 }
                ?: if (urlFull.startsWith(PROTOCOL_HTTPS)) 443 else 80
            attrs[ATTR_SERVER_PORT] = port.toLong()

            attrs[ATTR_CLIENT] = selfServiceName
            attrs[ATTR_SERVER] = props.serviceName.toString()

            props.serviceName?.let { attrs[ATTR_PEER_SERVICE] = it }

            // Размеры
            if (rq.method.name() !in METHODS_WITHOUT_BODY) {
                val reqSize = body.size.toLong()
                if (reqSize > 0) {
                    attrs[ATTR_HTTP_REQUEST_BODY_SIZE] = reqSize
                }
            }

            rs.headers.contentLength.takeIf { it != -1L }?.let {
                attrs[ATTR_HTTP_RESPONSE_BODY_SIZE] = it
            }

            if (isError) {
                attrs[ATTR_EXCEPTION_MESSAGE] = "HTTP ${rs.statusCode.value()}"
            }

            // Наполняем прямо в общую мапу без создания промежуточных Map
            fillRequestHeadersAttrs(rq, attrs)
            fillResponseHeadersAttrs(rs, attrs)

            // Фиксация успешного (или логического ошибочного, например 4xx/5xx) результата
            tracer.stop(
                ctx = ctx,
                status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_OK,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                forceExport = isSlow,
                attrs = attrs,
            )
            rs
        } catch (e: Exception) {
            val attrs = HashMap<String, Any>(8)
            attrs[ATTR_HTTP_REQUEST_METHOD] = rq.method.name()
            attrs[ATTR_URL_FULL] = rq.uri.toString()
            attrs[ATTR_CLIENT] = selfServiceName
            attrs[ATTR_SERVER] = props.serviceName.toString()

            fillRequestHeadersAttrs(rq, attrs)

            ctx.error = e
            tracer.stop(
                ctx = ctx,
                status = StatusCode.STATUS_CODE_ERROR,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                attrs = attrs,
            )
            throw e
        } finally {
            MDC.put(MDC_SOURCE, originalClient)
            MDC.put(MDC_TARGET, originalServer)
        }
    }

    private fun fillRequestHeadersAttrs(rq: HttpRequest, target: HashMap<String, Any>) {
        // У Spring HttpRequest.headers — это HttpHeaders (Map<String, List<String>>)
        val headers = rq.headers
        for (entry in headers.entries) {
            val name = entry.key
            val normalizedName = name.lowercase()

            if (normalizedName in OTEL_MAPPED_HEADERS) {
                continue
            }

            val key = "$PREFIX_HTTP_REQUEST_HEADER$normalizedName"
            target[key] = entry.value
        }
    }

    private fun fillResponseHeadersAttrs(rs: ClientHttpResponse, target: HashMap<String, Any>) {
        val headers = rs.headers
        for (entry in headers.entries) {
            val name = entry.key
            val normalizedName = name.lowercase()

            if (normalizedName in OTEL_MAPPED_HEADERS) {
                continue
            }

            val key = "$PREFIX_HTTP_RESPONSE_HEADER$normalizedName"
            target[key] = entry.value
        }
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

        const val PREFIX_CLIENT_SPAN = "CLIENT:"
        const val HOST_UNKNOWN = "unknown"
    }
}
