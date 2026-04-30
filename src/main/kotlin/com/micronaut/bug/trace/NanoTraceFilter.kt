package com.micronaut.bug.trace

import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_CLIENT
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_CLIENT_ADDRESS
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_HTTP_SLOW_REQUEST
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER_ADDRESS
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_SERVER_PORT
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_URL_FULL
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_URL_PATH
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_URL_QUERY
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_URL_SCHEME
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_USER_AGENT_ORIGINAL
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_TRACEPARENT
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.trace.NanoTracer.Companion.MASKED_VALUES
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_CLIENT
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SERVER
import com.micronaut.bug.trace.NanoTracer.Companion.METHODS_WITHOUT_BODY
import com.micronaut.bug.trace.NanoTracer.Companion.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.SENSITIVE_HEADERS
import com.micronaut.bug.trace.NanoTracer.Companion.TRACEPARENT_DELIMITER
import com.micronaut.bug.trace.NanoTracer.Companion.TRACEPARENT_PREFIX
import com.micronaut.bug.trace.NanoTracer.Companion.parseBaggage
import com.micronaut.bug.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Status
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.USER_AGENT
import org.springframework.web.filter.OncePerRequestFilter

class NanoTraceFilter(
    private val tracer: NanoTracer,
    private val traceProps: TraceProperties,
    private val selfServiceName: String,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        val startTimeNano = System.nanoTime()
        // 1. Извлекаем данные из заголовков

        val traceParent = rq.getHeader(HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null
        var isSampledByParent = true

        if (traceParent != null && traceParent.startsWith(TRACEPARENT_PREFIX)) {
            val parts = traceParent.split(TRACEPARENT_DELIMITER)
            if (parts.size >= 4) {
                traceId = parts[1]
                parentId = parts[2]
                // Проверяем флаг сэмплирования (последний бит)
                val flags = parts[3].toIntOrNull(16) ?: 0
                isSampledByParent = (flags and 0x01) == 1
            }
        }

        val sender = rq.getHeader(HEADER_X_SENDER) ?: DEFAULT_SENDER

        val baggage = mutableMapOf<String, String>()
        // Читаем стандартный багаж (если пришел)
        rq.getHeader(HEADER_BAGGAGE)?.let {
            baggage.putAll(parseBaggage(it))
        }

        // Читаем заголовки для проброски из конфига
        val propagationHeaders = traceProps.propagationHeaders
            .associateWith { rq.getHeader(it) }
            .filterValues { it != null }
            .mapKeys { it.key.lowercase() }

        val traceState = rq.getHeader(NanoTracer.HEADER_TRACESTATE)

        // Стартуем трейс, учитывая родителя
        val ctx = tracer.startTrace(
            name = "${rq.method} ${rq.requestURI}",
            remoteTraceId = traceId,
            remoteParentId = parentId,
            sampled = isSampledByParent,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            traceState = traceState,
        )

        MDC.put(MDC_CLIENT, sender)
        MDC.put(MDC_SERVER, selfServiceName)

        try {
            rs.setHeader(HEADER_TRACEPARENT, tracer.getTraceParent())

            chain.doFilter(rq, rs)
        } finally {

            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            val isSlow = durationMs >= traceProps.slowRequestThreshold.toMillis()

            // Если запрос медленный — пишем WARN лог, чтобы сразу подсветить проблему в Loki
            if (isSlow) {
                log.warn { "Slow request detected: ${rq.method} ${rq.requestURI} took ${durationMs}ms" }
            }

            val isError = rs.status >= ERROR_STATUS_THRESHOLD

            // Собираем базовые атрибуты по стандарту OTel
            val attrs = buildMap<String, Any> {
                put(ATTR_URL_FULL, getFullUri(rq))
                put(ATTR_URL_SCHEME, rq.scheme)
                put(ATTR_URL_PATH, rq.requestURI)
                rq.queryString?.let { put(ATTR_URL_QUERY, it) }

                put(ATTR_HTTP_REQUEST_METHOD, rq.method)
                rq.getHeader(USER_AGENT)?.let { put(ATTR_USER_AGENT_ORIGINAL, it) }

                if (rq.method !in METHODS_WITHOUT_BODY) {
                    val reqSize = rq.contentLength.toLong()
                    if (reqSize != -1L) {
                        put(ATTR_HTTP_REQUEST_BODY_SIZE, reqSize)
                    }
                }

                if (isSlow) {
                    put(ATTR_HTTP_SLOW_REQUEST, true)
                }

                // В HttpServletResponse размер можно вытащить через Content-Length
                rs.getHeader(HttpHeaders.CONTENT_LENGTH)
                    ?.toLongOrNull()
                    ?.let { put(ATTR_HTTP_RESPONSE_BODY_SIZE, it) }

                put(ATTR_CLIENT, sender)
                put(ATTR_SERVER, selfServiceName)

                put(ATTR_SERVER_ADDRESS, rq.serverName)
                put(ATTR_SERVER_PORT, rq.serverPort.toLong())
                put(ATTR_CLIENT_ADDRESS, rq.remoteAddr)
                put(ATTR_HTTP_RESPONSE_STATUS_CODE, rs.status.toLong())

                if (isError) {
                    put(ATTR_EXCEPTION_MESSAGE, "HTTP ${rs.status}")
                }

                putAll(getRequestHeadersAttrs(rq))
                putAll(getResponseHeadersAttrs(rs))
            }

            tracer.stop(
                ctx = ctx,
                status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                attrs = attrs,
                forceExport = isSlow || isError,
            )
            MDC.clear()
        }
    }


    // Хелперы для извлечения данных
    private fun getFullUri(rq: HttpServletRequest): String =
        rq.requestURL.let { if (rq.queryString != null) it.append(QUERY_MARKER).append(rq.queryString) else it }.toString()

    private fun getRequestHeadersAttrs(rq: HttpServletRequest): Map<String, List<String>> =
        rq.headerNames.asSequence()
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
                val normalizedName = name.lowercase()
                val key = "${PREFIX_HTTP_REQUEST_HEADER}$normalizedName"
                key to if (normalizedName in SENSITIVE_HEADERS) MASKED_VALUES else rq.getHeaders(name).toList()
            }

    private fun getResponseHeadersAttrs(rs: HttpServletResponse): Map<String, List<String>> =
        rs.headerNames.asSequence()
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
                val key = "${PREFIX_HTTP_RESPONSE_HEADER}${name.lowercase()}"
                key to rs.getHeaders(name).toList()
            }

    companion object {

        private const val DEFAULT_SENDER = "USER"
        private const val QUERY_MARKER = "?"
        private const val ERROR_STATUS_THRESHOLD = 400

        const val HEADER_API_KEY = "api-key"
    }
}
