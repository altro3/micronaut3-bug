package com.altro.common.client

import com.altro.common.trace.NanoTracer
import com.altro.common.trace.TraceUtil
import com.altro.common.trace.http.HttpTraceExtractor
import com.altro.common.trace.http.NanoTraceFilter
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import org.slf4j.MDC
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse
import kotlin.collections.iterator

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
            val path = HttpTraceExtractor.getFullUri(rq.uri)
            if (path.startsWith(LoggingInterceptor.SLASH)) "$basePrefix$path" else "$basePrefix/$path"
        }

        val methodStr = rq.method.name()
        val serviceNameStr = props.serviceName ?: HOST_UNKNOWN

        val userId = MDC.get(NanoTraceFilter.MDC_USER_ID)

        val span = tracer.startSpan(name = "$PREFIX_CLIENT_SPAN $methodStr $urlFull")

        tracer.getTraceParent()?.let { rq.headers.set(TraceUtil.HEADER_TRACEPARENT, it) }

        if (props.type == HttpClientProperties.ClientType.INTERNAL) {
            rq.headers.set(TraceUtil.HEADER_X_SENDER, selfServiceName)
            if (userId != null) {
                rq.headers.set(NanoTraceFilter.HEADER_USER_ID, userId)
            }
        }

        TraceUtil.formatBaggage(span.baggage)?.let { rq.headers.set(TraceUtil.HEADER_BAGGAGE, it) }

        span.traceState?.let {
            if (!rq.headers.containsKey(TraceUtil.HEADER_TRACESTATE)) {
                rq.headers.set(TraceUtil.HEADER_TRACESTATE, it)
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

        val originalClient = MDC.get(TraceUtil.MDC_SOURCE)
        val originalServer = MDC.get(TraceUtil.MDC_TARGET)

        return try {
            MDC.put(TraceUtil.MDC_SOURCE, selfServiceName)
            MDC.put(TraceUtil.MDC_TARGET, serviceNameStr)

            val rs = execution.execute(rq, body)

            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            val isSlow = durationMs >= props.log.slowThreshold.toMillis()

            if (isSlow) {
                log.warn { "Slow client response: $methodStr $urlFull took ${durationMs}ms (threshold: ${props.log.slowThreshold.toMillis()}ms)" }
            }

            val attrs = HashMap<String, Any>(32)

            HttpTraceExtractor.extractClientBaseAttributes(
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
                target = attrs,
                serviceName = props.serviceName,
                baseUrl = props.url,
            )
            HttpTraceExtractor.fillHeadersAttrs(rq.headers, TraceUtil.PREFIX_HTTP_REQUEST_HEADER, isSensitive = true, attrs)
            HttpTraceExtractor.fillHeadersAttrs(rs.headers, TraceUtil.PREFIX_HTTP_RESPONSE_HEADER, isSensitive = false, attrs)

            tracer.stop(
                span = span,
                status = if (rs.statusCode.isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_UNSET,
                kind = SpanKind.SPAN_KIND_CLIENT,
                forceExport = isSlow,
                attrs = attrs,
            )
            rs
        } catch (e: Exception) {
            val attrs = HashMap<String, Any>(16)
            HttpTraceExtractor.extractClientBaseAttributes(
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
                target = attrs,
                serviceName = props.serviceName,
                baseUrl = props.url,
            )
            HttpTraceExtractor.fillHeadersAttrs(rq.headers, TraceUtil.PREFIX_HTTP_REQUEST_HEADER, isSensitive = true, attrs)

            span.error = e
            tracer.stop(
                span = span,
                status = StatusCode.STATUS_CODE_ERROR,
                kind = SpanKind.SPAN_KIND_CLIENT,
                attrs = attrs,
            )
            throw e
        } finally {
            MDC.put(TraceUtil.MDC_SOURCE, originalClient)
            MDC.put(TraceUtil.MDC_TARGET, originalServer)
        }
    }

    companion object {
        const val PREFIX_CLIENT_SPAN = "CLIENT:"
        const val HOST_UNKNOWN = "unknown"
    }
}