package com.micronaut.bug.trace

import com.micronaut.bug.trace.HttpTraceExtractor.extractBaseAttributes
import com.micronaut.bug.trace.HttpTraceExtractor.fillRequestHeadersAttrs
import com.micronaut.bug.trace.HttpTraceExtractor.fillResponseHeadersAttrs
import com.micronaut.bug.trace.TraceUtil.ATTR_CLIENT
import com.micronaut.bug.trace.TraceUtil.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.TraceUtil.ATTR_SERVER
import com.micronaut.bug.trace.TraceUtil.HEADER_BAGGAGE
import com.micronaut.bug.trace.TraceUtil.HEADER_TRACEPARENT
import com.micronaut.bug.trace.TraceUtil.HEADER_TRACESTATE
import com.micronaut.bug.trace.TraceUtil.HEADER_X_SENDER
import com.micronaut.bug.trace.TraceUtil.MDC_SOURCE
import com.micronaut.bug.trace.TraceUtil.MDC_TARGET
import com.micronaut.bug.trace.TraceUtil.TRACEPARENT_PREFIX
import com.micronaut.bug.trace.TraceUtil.parseBaggage
import com.micronaut.bug.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Status
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
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

        if (withActuator && rq.requestURI.startsWith(rq.contextPath + PATH_ACTUATOR)) {
            chain.doFilter(rq, rs)
            return
        }

        val userId = rq.getHeader(HEADER_USER_ID)
        val traceParent = rq.getHeader(HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null
        var isSampledByParent = true

        if (traceParent != null && traceParent.length == 55 && traceParent.startsWith(TRACEPARENT_PREFIX)) {
            val firstDash = traceParent.indexOf('-', 3)
            val secondDash = traceParent.indexOf('-', firstDash + 1)

            if (firstDash == 35 && secondDash == 52) {
                traceId = traceParent.substring(3, firstDash)
                parentId = traceParent.substring(firstDash + 1, secondDash)

                val f2 = traceParent[54]
                isSampledByParent = (f2 == '1' || f2 == '3' || f2 == '5' || f2 == '7' || f2 == '9' || f2 == 'b' || f2 == 'd' || f2 == 'f')
            }
        }

        val sender = rq.getHeader(HEADER_X_SENDER) ?: DEFAULT_SENDER

        val baggage = mutableMapOf<String, String>()
        rq.getHeader(HEADER_BAGGAGE)?.let { header ->
            parseBaggage(header)?.let { baggage.putAll(it) }
        }

        val propagationHeaders = HashMap<String, String>()
        val propKeys = traceProps.propagationHeaders

        for (key in propKeys) {
            val value = rq.getHeader(key)
            if (value != null) {
                propagationHeaders[key.lowercase()] = value
            }
        }

        val traceState = rq.getHeader(HEADER_TRACESTATE)

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
            MDC.put(MDC_SOURCE, sender)
            MDC.put(MDC_TARGET, selfServiceName)

            rs.setHeader(HEADER_TRACEPARENT, tracer.getTraceParent())
            chain.doFilter(rq, rs)
        } finally {
            try {
                val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
                val isSlow = durationMs >= traceProps.slowRequestThreshold.toMillis()
                val isError = rs.status >= ERROR_STATUS_THRESHOLD
                val attrs = HashMap<String, Any>(32)

                MDC.get(MDC_USER_ID)?.let { attrs[ATTR_USER_ID] = it }
                attrs[ATTR_CLIENT] = sender
                attrs[ATTR_SERVER] = selfServiceName

                if (isError) attrs[ATTR_EXCEPTION_MESSAGE] = "HTTP ${rs.status}"

                extractBaseAttributes(rq, rs, attrs, isSlow)
                fillRequestHeadersAttrs(rq, attrs)
                fillResponseHeadersAttrs(rs, attrs)

                tracer.stop(
                    span = curSpan,
                    status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                    attrs = attrs,
                    forceExport = isSlow || isError,
                )
            } catch (e: Throwable) {
                log.error(e) { "Tracer failed to extract HTTP trace attributes" }
            } finally {
                tracer.clearThreadSpan()
                MDC.remove(MDC_USER_ID)
                MDC.remove(MDC_SOURCE)
                MDC.remove(MDC_TARGET)
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
