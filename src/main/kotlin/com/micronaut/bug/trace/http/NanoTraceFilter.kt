package com.micronaut.bug.trace.http

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.TraceUtil
import com.micronaut.bug.trace.config.TraceProperties.TraceHttpProperties
import com.micronaut.bug.trace.http.HttpTraceExtractor.extractServerBaseAttributes
import com.micronaut.bug.trace.http.HttpTraceExtractor.fillRequestHeadersAttrs
import com.micronaut.bug.trace.http.HttpTraceExtractor.fillResponseHeadersAttrs
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Status
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.util.ClassUtils
import org.springframework.web.filter.OncePerRequestFilter

class NanoTraceFilter(
    private val tracer: NanoTracer,
    private val traceProps: TraceHttpProperties,
    private val selfServiceName: String,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    private val withActuator: Boolean = ClassUtils.isPresent("org.springframework.boot.actuate.autoconfigure.endpoint.web.WebEndpointProperties", null)
    private val slowRequestThresholdMs = traceProps.slowRequestThreshold.toMillis()

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        val startTimeNano = System.nanoTime()

        if (withActuator && rq.requestURI.startsWith(rq.contextPath + PATH_ACTUATOR)) {
            chain.doFilter(rq, rs)
            return
        }

        val userId = rq.getHeader(HEADER_USER_ID)
        val traceParent = rq.getHeader(TraceUtil.HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null
        var isSampledByParent = true

        if (traceParent != null && traceParent.length == 55 && traceParent.startsWith(TraceUtil.TRACEPARENT_PREFIX)) {
            val firstDash = traceParent.indexOf('-', 3)
            val secondDash = traceParent.indexOf('-', firstDash + 1)

            if (firstDash == 35 && secondDash == 52) {
                traceId = traceParent.substring(3, firstDash)
                parentId = traceParent.substring(firstDash + 1, secondDash)

                val f2 = traceParent[54]
                isSampledByParent = (f2 == '1' || f2 == '3' || f2 == '5' || f2 == '7' || f2 == '9' || f2 == 'b' || f2 == 'd' || f2 == 'f')
            }
        }

        val sender = rq.getHeader(TraceUtil.HEADER_X_SENDER) ?: DEFAULT_SENDER

        val baggage = mutableMapOf<String, String>()
        rq.getHeader(TraceUtil.HEADER_BAGGAGE)?.let { header ->
            TraceUtil.parseBaggage(header)?.let { baggage.putAll(it) }
        }

        val propagationHeaders = HashMap<String, String>()
        val propKeys = traceProps.propagationHeaders

        for (key in propKeys) {
            val value = rq.getHeader(key)
            if (value != null) {
                propagationHeaders[key.lowercase()] = value
            }
        }

        val traceState = rq.getHeader(TraceUtil.HEADER_TRACESTATE)

        val curSpan = tracer.startTrace(
            name = "${rq.method} ${rq.requestURI}",
            remoteTraceId = traceId,
            remoteParentId = parentId,
            sampled = isSampledByParent,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            traceState = traceState,
        )

        try {
            if (userId != null) {
                MDC.put(MDC_USER_ID, userId)
            }
            MDC.put(TraceUtil.MDC_SOURCE, sender)
            MDC.put(TraceUtil.MDC_TARGET, selfServiceName)

            rs.setHeader(TraceUtil.HEADER_TRACEPARENT, tracer.getTraceParent())
            chain.doFilter(rq, rs)
        } finally {
            try {
                val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
                val isSlow = durationMs >= slowRequestThresholdMs
                val isError = rs.status >= ERROR_STATUS_THRESHOLD
                val attrs = HashMap<String, Any>(32)

                MDC.get(MDC_USER_ID)?.let { attrs[ATTR_USER_ID] = it }
                attrs[TraceUtil.ATTR_CLIENT] = sender
                attrs[TraceUtil.ATTR_SERVER] = selfServiceName

                if (isError) attrs[TraceUtil.ATTR_EXCEPTION_MESSAGE] = "HTTP ${rs.status}"

                extractServerBaseAttributes(
                    methodStr = rq.method,
                    scheme = rq.scheme,
                    path = rq.requestURI,
                    queryString = rq.queryString,
                    urlFull = HttpTraceExtractor.getFullUri(rq),
                    userAgent = rq.getHeader(HttpHeaders.USER_AGENT),
                    remoteAddr = rq.remoteAddr,
                    serverName = rq.serverName,
                    serverPort = rq.serverPort,
                    reqBodySize = rq.contentLength.toLong().takeIf { it != -1L } ?: 0L,
                    resStatusCode = rs.status,
                    resContentLength = rs.getHeader(HttpHeaders.CONTENT_LENGTH)?.toLongOrNull(),
                    isSlow = isSlow,
                    target = attrs
                )
                fillRequestHeadersAttrs(rq, attrs)
                fillResponseHeadersAttrs(rs, attrs)

                tracer.stop(
                    span = curSpan,
                    status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_UNSET,
                    attrs = attrs,
                    forceExport = isSlow || isError,
                )
            } catch (e: Throwable) {
                log.error(e) { "Tracer failed to extract HTTP trace attributes" }
            } finally {
                tracer.clearThreadSpan()
                MDC.remove(MDC_USER_ID)
                MDC.remove(TraceUtil.MDC_SOURCE)
                MDC.remove(TraceUtil.MDC_TARGET)
            }
        }
    }

    companion object {
        private const val PATH_ACTUATOR = "/actuator"
        private const val DEFAULT_SENDER = "USER"
        private const val ERROR_STATUS_THRESHOLD = 400

        const val HEADER_API_KEY = "api-key"
        const val HEADER_USER_ID = "x-user-id"
        const val ATTR_USER_ID = "user.id"
        const val MDC_USER_ID = "userId"
    }
}