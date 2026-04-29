package com.micronaut.bug.trace

import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_TRACEPARENT
import com.micronaut.bug.trace.NanoTracer.Companion.HEADER_X_SENDER
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_CLIENT
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SERVER
import com.micronaut.bug.trace.NanoTracer.Companion.TRACEPARENT_DELIMITER
import com.micronaut.bug.trace.NanoTracer.Companion.TRACEPARENT_PREFIX
import com.micronaut.bug.trace.NanoTracer.Companion.parseBaggage
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_CLIENT
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_CLIENT_ADDRESS
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER_ADDRESS
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_SERVER_PORT
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_URL_FULL
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_URL_PATH
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_URL_QUERY
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_URL_SCHEME
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_USER_AGENT_ORIGINAL
import com.micronaut.bug.trace.TempoExporter.Companion.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.config.TraceProperties
import io.opentelemetry.proto.trace.v1.Status
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.USER_AGENT
import org.springframework.http.HttpMethod
import org.springframework.web.filter.OncePerRequestFilter

class NanoTraceFilter(
    private val tracer: NanoTracer,
    private val traceProps: TraceProperties,
    private val selfServiceName: String,
) : OncePerRequestFilter() {

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        // 1. Извлекаем данные из заголовков

        val traceParent = rq.getHeader(HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null

        if (traceParent != null && traceParent.startsWith(TRACEPARENT_PREFIX)) {
            val parts = traceParent.split(TRACEPARENT_DELIMITER)
            if (parts.size >= 3) {
                traceId = parts[1]
                parentId = parts[2]
            }
        }

        val sender = rq.getHeader(HEADER_X_SENDER) ?: DEFAULT_SENDER

        val baggage = mutableMapOf<String, String>()
        // 1. Читаем стандартный багаж (если пришел)
        rq.getHeader(HEADER_BAGGAGE)?.let {
            baggage.putAll(parseBaggage(it))
        }

        // 2. Читаем заголовки для проброски из конфига
        val propagationHeaders = traceProps.propagationHeaders
            .associateWith { rq.getHeader(it) }
            .filterValues { it != null }
            .mapKeys { it.key.lowercase() }

        // 2. Стартуем трейс, учитывая родителя
        val ctx = tracer.startTrace(
            name = "${rq.method} ${rq.requestURI}",
            remoteTraceId = traceId,
            remoteParentId = parentId,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
        )

        MDC.put(MDC_CLIENT, sender)
        MDC.put(MDC_SERVER, selfServiceName)

        try {
            chain.doFilter(rq, rs)
        } finally {

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
        private const val MASK = "***"
        private const val ERROR_STATUS_THRESHOLD = 400

        const val HEADER_API_KEY = "api-key"

        val METHODS_WITHOUT_BODY = setOf(
            HttpMethod.GET.name(),
            HttpMethod.HEAD.name(),
            HttpMethod.OPTIONS.name(),
            HttpMethod.DELETE.name(),
            HttpMethod.TRACE.name(),
        )

        /**
         * Список для маскировки чувствительных заголовков в формате OTel (string[]).
         */
        private val MASKED_VALUES = listOf(MASK)

        private val SENSITIVE_HEADERS = setOf(
            HttpHeaders.AUTHORIZATION.lowercase(),
            HttpHeaders.COOKIE.lowercase(),
            HttpHeaders.SET_COOKIE.lowercase(),
            HEADER_API_KEY.lowercase(),
        )
    }
}
