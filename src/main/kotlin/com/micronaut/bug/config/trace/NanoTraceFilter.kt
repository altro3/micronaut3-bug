package com.micronaut.bug.config.trace

import com.micronaut.bug.client.HttpClientConst
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_EXT_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_RQ_ID
import com.micronaut.bug.config.trace.NanoTracer.Companion.HEADER_X_SENDER
import io.opentelemetry.proto.trace.v1.Status
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.web.filter.OncePerRequestFilter

class NanoTraceFilter(
    private val tracer: NanoTracer,
) : OncePerRequestFilter() {

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        // 1. Извлекаем данные из заголовков
        val traceId = rq.getHeader(HEADER_X_RQ_ID)?.takeIf { it.isNotBlank() }
        val parentId = rq.getHeader(HEADER_EXT_RQ_ID)?.takeIf { it.isNotBlank() } // ID спана соседа
        val sender = rq.getHeader(HEADER_X_SENDER) ?: DEFAULT_SENDER

        // 2. Стартуем трейс, учитывая родителя
        val ctx = tracer.startTrace(
            name = "${rq.method} ${rq.requestURI}",
            remoteTraceId = traceId,
            remoteParentId = parentId,
        )

        MDC.put(NanoTracer.MDC_CLIENT, sender)

        try {
            chain.doFilter(rq, rs)
        } finally {

            val isError = rs.status >= ERROR_STATUS_THRESHOLD

            tracer.stop(
                ctx = ctx,
                status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                attrs = mapOf(
                    TempoExporter.ATTR_HTTP_METHOD to rq.method,
                    TempoExporter.ATTR_HTTP_URL to getFullUri(rq),
                    TempoExporter.ATTR_CLIENT_ID to sender,
                    TempoExporter.ATTR_HTTP_STATUS_CODE to rs.status,
                    TempoExporter.ATTR_RQ_HEADERS to getHeadersMap(rq).toString(),
                    TempoExporter.ATTR_RS_HEADERS to getResponseHeaders(rs).toString(),
                ),
            )
            MDC.clear()
        }
    }


    // Хелперы для извлечения данных (можно вынести в утилиты)
    private fun getFullUri(rq: HttpServletRequest): String =
        rq.requestURL.let { if (rq.queryString != null) it.append(QUERY_MARKER).append(rq.queryString) else it }.toString()

    private fun getHeadersMap(rq: HttpServletRequest): Map<String, String> =
        rq.headerNames.asSequence()
            .associateWith { name ->
                if (name.lowercase() in SENSITIVE_HEADERS) MASK else rq.getHeader(name)
            }

    private fun getResponseHeaders(rs: HttpServletResponse): Map<String, String> =
        rs.headerNames.associateWith { rs.getHeader(it) }

    companion object {

        private const val DEFAULT_SENDER = "USER"
        private const val QUERY_MARKER = "?"
        private const val MASK = "***"
        private const val ERROR_STATUS_THRESHOLD = 400

        private val SENSITIVE_HEADERS = setOf(
            HttpHeaders.AUTHORIZATION,
            HttpHeaders.COOKIE,
            HttpHeaders.SET_COOKIE,
            HttpClientConst.HEADER_API_KEY,
        )
    }
}