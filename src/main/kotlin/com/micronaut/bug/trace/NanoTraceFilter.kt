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
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_SOURCE
import com.micronaut.bug.trace.NanoTracer.Companion.MDC_TARGET
import com.micronaut.bug.trace.NanoTracer.Companion.METHODS_WITHOUT_BODY
import com.micronaut.bug.trace.NanoTracer.Companion.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.NanoTracer.Companion.SENSITIVE_HEADERS
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
import org.springframework.util.ClassUtils
import org.springframework.web.filter.OncePerRequestFilter

class NanoTraceFilter(
    private val tracer: NanoTracer,
    private val traceProps: TraceProperties,
    private val selfServiceName: String,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    private val withActuator: Boolean = ClassUtils.isPresent("org.springframework.boot.actuate.autoconfigure.endpoint.web.WebEndpointProperties", null)

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {

        val startTimeNano = System.nanoTime()
        // Пропускаем Actuator-эндпоинты, если это указано в настройках
        if (withActuator && rq.requestURI.startsWith(rq.contextPath + PATH_ACTUATOR)) {
            chain.doFilter(rq, rs)
            return
        }

        val userId = rq.getHeader(HEADER_USER_ID)

        // 1. Безаллокационный разбор W3C traceparent
        val traceParent = rq.getHeader(HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null
        var isSampledByParent = true

        if (traceParent != null && traceParent.startsWith(TRACEPARENT_PREFIX)) {
            val firstDash = traceParent.indexOf('-', 3)
            val secondDash = traceParent.indexOf('-', firstDash + 1)

            if (firstDash != -1 && secondDash != -1) {
                traceId = traceParent.substring(3, firstDash)
                parentId = traceParent.substring(firstDash + 1, secondDash)

                val flagsStr = traceParent.substring(secondDash + 1)
                if (flagsStr.length >= 2) {
                    val flags = flagsStr.toIntOrNull(16) ?: 0
                    isSampledByParent = (flags and 0x01) == 1
                }
            }
        }

        val sender = rq.getHeader(HEADER_X_SENDER) ?: DEFAULT_SENDER

        val baggage = mutableMapOf<String, String>()
        rq.getHeader(HEADER_BAGGAGE)?.let { header ->
            parseBaggage(header)?.let { baggage.putAll(it) }
        }

        val propagationHeaders = HashMap<String, String>()
        traceProps.propagationHeaders.forEach { key ->
            val value = rq.getHeader(key)
            if (value != null) {
                propagationHeaders[key.lowercase()] = value
            }
        }

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

        if (userId != null) {
            MDC.put(MDC_USER_ID, userId)
        }
        MDC.put(MDC_SOURCE, sender)
        MDC.put(MDC_TARGET, selfServiceName)

        try {
            rs.setHeader(HEADER_TRACEPARENT, tracer.getTraceParent())
            chain.doFilter(rq, rs)
        } finally {
            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            val isSlow = durationMs >= traceProps.slowRequestThreshold.toMillis()

            if (isSlow) {
                log.warn { "Slow request detected: ${rq.method} ${rq.requestURI} took ${durationMs}ms" }
            }

            val isError = rs.status >= ERROR_STATUS_THRESHOLD

            // 3. Прямое наполнение HashMap вместо тяжелого buildMap
            val attrs = HashMap<String, Any>(32)

            MDC.get(MDC_USER_ID)?.let { attrs[ATTR_USER_ID] = it }
            attrs[ATTR_URL_FULL] = getFullUri(rq)
            attrs[ATTR_URL_SCHEME] = rq.scheme
            attrs[ATTR_URL_PATH] = rq.requestURI
            rq.queryString?.let { attrs[ATTR_URL_QUERY] = it }

            attrs[ATTR_HTTP_REQUEST_METHOD] = rq.method
            rq.getHeader(USER_AGENT)?.let { attrs[ATTR_USER_AGENT_ORIGINAL] = it }

            if (rq.method !in METHODS_WITHOUT_BODY) {
                val reqSize = rq.contentLength.toLong()
                if (reqSize != -1L) {
                    attrs[ATTR_HTTP_REQUEST_BODY_SIZE] = reqSize
                }
            }

            if (isSlow) {
                attrs[ATTR_HTTP_SLOW_REQUEST] = true
            }

            rs.getHeader(HttpHeaders.CONTENT_LENGTH)
                ?.toLongOrNull()
                ?.let { attrs[ATTR_HTTP_RESPONSE_BODY_SIZE] = it }

            attrs[ATTR_CLIENT] = sender
            attrs[ATTR_SERVER] = selfServiceName

            attrs[ATTR_SERVER_ADDRESS] = rq.serverName
            attrs[ATTR_SERVER_PORT] = rq.serverPort.toLong()
            attrs[ATTR_CLIENT_ADDRESS] = rq.remoteAddr
            attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = rs.status.toLong()

            if (isError) {
                attrs[ATTR_EXCEPTION_MESSAGE] = "HTTP ${rs.status}"
            }

            // 4. Наполняем мапу напрямую без создания лишних коллекций
            fillRequestHeadersAttrs(rq, attrs)
            fillResponseHeadersAttrs(rs, attrs)

            tracer.stop(
                ctx = ctx,
                status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                attrs = attrs,
                forceExport = isSlow || isError,
            )
            MDC.clear()
        }
    }

    private fun getFullUri(rq: HttpServletRequest): String =
        rq.requestURL.let { if (rq.queryString != null) it.append(QUERY_MARKER).append(rq.queryString) else it }.toString()

    // 5. Потоковое наполнение атрибутов без аллокаций Sequence
    private fun fillRequestHeadersAttrs(rq: HttpServletRequest, target: HashMap<String, Any>) {
        val names = rq.headerNames ?: return
        while (names.hasMoreElements()) {
            val name = names.nextElement()
            val normalizedName = name.lowercase()

            if (normalizedName in OTEL_MAPPED_HEADERS) continue

            val key = "$PREFIX_HTTP_REQUEST_HEADER$normalizedName"

            if (normalizedName in SENSITIVE_HEADERS) {
                target[key] = MASKED_VALUES
            } else {
                val headersEnum = rq.getHeaders(name)
                val valuesList = ArrayList<String>()
                while (headersEnum.hasMoreElements()) {
                    valuesList.add(headersEnum.nextElement())
                }
                target[key] = valuesList
            }
        }
    }

    private fun fillResponseHeadersAttrs(rs: HttpServletResponse, target: HashMap<String, Any>) {
        val names = rs.headerNames ?: return
        for (name in names) {
            val normalizedName = name.lowercase()
            if (normalizedName in OTEL_MAPPED_HEADERS) continue

            val key = "$PREFIX_HTTP_RESPONSE_HEADER$normalizedName"
            val headers = rs.getHeaders(name)
            val valuesList = ArrayList<String>(headers.size)
            for (value in headers) {
                valuesList.add(value)
            }
            target[key] = valuesList
        }
    }

    companion object {

        private const val PATH_ACTUATOR = "/actuator"

        private const val DEFAULT_SENDER = "USER"
        private const val QUERY_MARKER = "?"
        private const val ERROR_STATUS_THRESHOLD = 400

        const val HEADER_API_KEY = "api-key"

        const val HEADER_USER_ID = "x-user-id" // Или твой заголовок
        const val ATTR_USER_ID = "user.id"
        const val MDC_USER_ID = "userId"
    }
}
