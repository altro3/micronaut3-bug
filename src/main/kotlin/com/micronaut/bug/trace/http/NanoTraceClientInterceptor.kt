package com.micronaut.bug.trace.http

import com.micronaut.bug.client.HttpClientProperties
import com.micronaut.bug.client.LoggingInterceptor
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.TraceUtil.HEADER_BAGGAGE
import com.micronaut.bug.trace.TraceUtil.HEADER_TRACEPARENT
import com.micronaut.bug.trace.TraceUtil.HEADER_TRACESTATE
import com.micronaut.bug.trace.TraceUtil.HEADER_X_SENDER
import com.micronaut.bug.trace.TraceUtil.MDC_SOURCE
import com.micronaut.bug.trace.TraceUtil.MDC_TARGET
import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.TraceUtil.formatBaggage
import com.micronaut.bug.trace.http.HttpTraceExtractor.extractClientBaseAttributes
import com.micronaut.bug.trace.http.HttpTraceExtractor.fillHeadersAttrs
import com.micronaut.bug.trace.http.HttpTraceExtractor.getFullUri
import com.micronaut.bug.trace.http.NanoTraceFilter.Companion.MDC_USER_ID
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status
import org.slf4j.MDC
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse

class NanoTraceClientInterceptor(
    private val tracer: NanoTracer,
    private val selfServiceName: String,
    private val props: HttpClientProperties,
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}
    private val basePrefix: String = props.url.toString().removeSuffix(LoggingInterceptor.SLASH)

    override fun intercept(rq: HttpRequest, body: ByteArray, execution: ClientHttpRequestExecution): ClientHttpResponse {
        val startTimeNano = System.nanoTime()
        val urlFull = if (rq.uri.isAbsolute) {
            rq.uri.toString()
        } else {
            val path = getFullUri(rq.uri)
            if (path.startsWith(LoggingInterceptor.SLASH)) "$basePrefix$path" else "$basePrefix/$path"
        }

        val methodStr = rq.method.name()
        val serviceNameStr = props.serviceName ?: HOST_UNKNOWN

        val userId = MDC.get(MDC_USER_ID)

        val span = tracer.startSpan(name = "$PREFIX_CLIENT_SPAN $methodStr $urlFull")

        tracer.getTraceParent()?.let { rq.headers.set(HEADER_TRACEPARENT, it) }

        if (props.type == HttpClientProperties.ClientType.INTERNAL) {
            rq.headers.set(HEADER_X_SENDER, selfServiceName)
            if (userId != null) {
                rq.headers.set(NanoTraceFilter.HEADER_USER_ID, userId)
            }
        }

        formatBaggage(span.baggage)?.let { rq.headers.set(HEADER_BAGGAGE, it) }

        span.traceState?.let {
            if (!rq.headers.containsKey(HEADER_TRACESTATE)) {
                rq.headers.set(HEADER_TRACESTATE, it)
            }
        }

        val propHeaders = span.propagationHeaders
        if (!propHeaders.isNullOrEmpty()) {
            val headers = rq.headers
            for ((key, value) in propHeaders) {
                if (!headers.containsKey(key)) {
                    headers.set(key, value)
                }
            }
        }

        val originalClient = MDC.get(MDC_SOURCE)
        val originalServer = MDC.get(MDC_TARGET)

        return try {
            MDC.put(MDC_SOURCE, selfServiceName)
            MDC.put(MDC_TARGET, serviceNameStr)

            val rs = execution.execute(rq, body)

            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            val isSlow = durationMs >= props.log.slowThreshold.toMillis()

            if (isSlow) {
                log.warn { "Slow client response: $methodStr $urlFull took ${durationMs}ms (threshold: ${props.log.slowThreshold.toMillis()}ms)" }
            }

            val attrs = HashMap<String, Any>(32)

            extractClientBaseAttributes(
                uri = rq.uri,
                bodySize = body.size.toLong(),
                statusCode = rs.statusCode.value(),
                contentLength = rs.headers.contentLength,
                isError = rs.statusCode.isError,
                urlFull = urlFull,
                methodStr = methodStr,
                selfServiceName = selfServiceName,
                serviceNameStr = serviceNameStr,
                userId = userId,
                props = props,
                target = attrs,
            )
            fillHeadersAttrs(rq.headers, PREFIX_HTTP_REQUEST_HEADER, isSensitive = true, attrs)
            fillHeadersAttrs(rs.headers, PREFIX_HTTP_RESPONSE_HEADER, isSensitive = false, attrs)

            tracer.stop(
                span = span,
                status = if (rs.statusCode.isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_UNSET,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                forceExport = isSlow,
                attrs = attrs,
            )
            rs
        } catch (e: Exception) {
            val attrs = HashMap<String, Any>(16)
            extractClientBaseAttributes(
                uri = rq.uri,
                bodySize = body.size.toLong(),
                statusCode = null,
                contentLength = null,
                isError = true,
                urlFull = urlFull,
                methodStr = methodStr,
                selfServiceName = selfServiceName,
                serviceNameStr = serviceNameStr,
                userId = userId,
                props = props,
                target = attrs
            )
            fillHeadersAttrs(rq.headers, PREFIX_HTTP_REQUEST_HEADER, isSensitive = true, attrs)

            span.error = e
            tracer.stop(
                span = span,
                status = Status.StatusCode.STATUS_CODE_ERROR,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                attrs = attrs,
            )
            throw e
        } finally {
            MDC.put(MDC_SOURCE, originalClient)
            MDC.put(MDC_TARGET, originalServer)
        }
    }

    companion object {
        const val PREFIX_CLIENT_SPAN = "CLIENT:"
        const val HOST_UNKNOWN = "unknown"
    }
}
