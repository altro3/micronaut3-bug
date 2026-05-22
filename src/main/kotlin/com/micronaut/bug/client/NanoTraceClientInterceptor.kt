package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.LoggingInterceptor.Companion.SLASH
import com.micronaut.bug.trace.NanoTraceFilter.Companion.ATTR_USER_ID
import com.micronaut.bug.trace.NanoTraceFilter.Companion.HEADER_USER_ID
import com.micronaut.bug.trace.NanoTraceFilter.Companion.MDC_USER_ID
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

class NanoTraceClientInterceptor(
    private val tracer: NanoTracer,
    private val selfServiceName: String,
    private val props: HttpClientProperties,
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}

    private val basePrefix: String = props.url.toString().removeSuffix(SLASH)

    override fun intercept(rq: HttpRequest, body: ByteArray, execution: ClientHttpRequestExecution): ClientHttpResponse {
        val startTimeNano = System.nanoTime()
        val urlFull = if (rq.uri.isAbsolute) {
            rq.uri.toString()
        } else {
            val path = getFullUri(rq)
            if (path.startsWith(SLASH)) "$basePrefix$path" else "$basePrefix/$path"
        }

        val methodStr = rq.method.name()

        val ctx = tracer.startSpan(name = "$PREFIX_CLIENT_SPAN $methodStr $urlFull")

        tracer.getTraceParent()?.let {
            rq.headers.set(HEADER_TRACEPARENT, it)
        }

        if (props.type == INTERNAL) {
            rq.headers.set(HEADER_X_SENDER, selfServiceName)
            MDC.get(MDC_USER_ID)?.let {
                rq.headers.set(HEADER_USER_ID, it)
            }
        }

        formatBaggage(ctx.baggage)?.let {
            rq.headers.set(HEADER_BAGGAGE, it)
        }

        ctx.traceState?.let {
            if (!rq.headers.containsKey(NanoTracer.HEADER_TRACESTATE)) {
                rq.headers.set(NanoTracer.HEADER_TRACESTATE, it)
            }
        }

        val propHeaders = ctx.propagationHeaders
        if (propHeaders != null && propHeaders.isNotEmpty()) {
            val headers = rq.headers
            for ((key, value) in propHeaders) {
                if (!headers.containsKey(key)) {
                    headers.set(key, value)
                }
            }
        }

        val originalClient = MDC.get(MDC_SOURCE)
        val originalServer = MDC.get(MDC_TARGET)

        val serviceNameStr = props.serviceName ?: HOST_UNKNOWN

        return try {
            val rs = execution.execute(rq, body)

            val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
            val isSlow = durationMs >= props.log.slowThreshold.toMillis()

            if (isSlow) {
                log.warn { "Slow client response: $methodStr $urlFull took ${durationMs}ms (threshold: ${props.log.slowThreshold.toMillis()}ms) [extRqId: ${MDC.get(MDC_SPAN_ID)}]" }
            }

            val statusCode = rs.statusCode
            val isError = statusCode.isError

            MDC.put(MDC_SOURCE, selfServiceName)
            MDC.put(MDC_TARGET, serviceNameStr)

            val attrs = HashMap<String, Any>(32)
            MDC.get(MDC_USER_ID)?.let { attrs[ATTR_USER_ID] = it }
            attrs[ATTR_HTTP_REQUEST_METHOD] = methodStr
            attrs[ATTR_URL_FULL] = urlFull
            attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = statusCode.value().toLong()

            val host = rq.uri.host ?: props.url.host ?: HOST_UNKNOWN
            attrs[ATTR_SERVER_ADDRESS] = host
            val port = rq.uri.port.takeIf { it != -1 }
                ?: props.url.port.takeIf { it != -1 }
                ?: if (urlFull.startsWith(PROTOCOL_HTTPS)) 443 else 80
            attrs[ATTR_SERVER_PORT] = port.toLong()

            attrs[ATTR_CLIENT] = selfServiceName
            attrs[ATTR_SERVER] = serviceNameStr
            attrs[ATTR_PEER_SERVICE] = props.serviceName ?: host

            if (methodStr !in METHODS_WITHOUT_BODY) {
                val reqSize = body.size.toLong()
                if (reqSize > 0) {
                    attrs[ATTR_HTTP_REQUEST_BODY_SIZE] = reqSize
                }
            }

            rs.headers.contentLength.takeIf { it != -1L }?.let {
                attrs[ATTR_HTTP_RESPONSE_BODY_SIZE] = it
            }

            if (isError) {
                attrs[ATTR_EXCEPTION_MESSAGE] = "HTTP ${statusCode.value()}"
            }

            fillRequestHeadersAttrs(rq, attrs)
            fillResponseHeadersAttrs(rs, attrs)

            tracer.stop(
                span = ctx,
                status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_OK,
                kind = Span.SpanKind.SPAN_KIND_CLIENT,
                forceExport = isSlow,
                attrs = attrs,
            )
            rs
        } catch (e: Exception) {
            val attrs = HashMap<String, Any>(8)
            MDC.get(MDC_USER_ID)?.let { attrs[ATTR_USER_ID] = it }
            attrs[ATTR_HTTP_REQUEST_METHOD] = methodStr
            attrs[ATTR_URL_FULL] = rq.uri.toString()
            attrs[ATTR_CLIENT] = selfServiceName
            attrs[ATTR_SERVER] = serviceNameStr

            fillRequestHeadersAttrs(rq, attrs)

            ctx.error = e
            tracer.stop(
                span = ctx,
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
        val entries = rq.headers.entries
        for (entry in entries) {
            val name = entry.key
            val normalizedName = name.lowercase()
            if (normalizedName in OTEL_MAPPED_HEADERS) continue

            val key = "$PREFIX_HTTP_REQUEST_HEADER$normalizedName"
            val values = entry.value
            if (values.size == 1) {
                target[key] = values[0]
            } else if (values.size > 1) {
                target[key] = values
            }
        }
    }

    private fun fillResponseHeadersAttrs(rs: ClientHttpResponse, target: HashMap<String, Any>) {
        val entries = rs.headers.entries
        for (entry in entries) {
            val name = entry.key
            val normalizedName = name.lowercase()
            if (normalizedName in OTEL_MAPPED_HEADERS) continue

            val key = "$PREFIX_HTTP_RESPONSE_HEADER$normalizedName"
            val values = entry.value
            if (values.size == 1) {
                target[key] = values[0]
            } else if (values.size > 1) {
                target[key] = values
            }
        }
    }

    private fun getFullUri(rq: HttpRequest): String {
        val uri = rq.uri
        val query = uri.query
        return if (query != null) "${uri.path}?$query" else uri.path
    }

    companion object {
        const val PROTOCOL_HTTP = "http"
        const val PROTOCOL_HTTPS = "https"
        const val PREFIX_CLIENT_SPAN = "CLIENT:"
        const val HOST_UNKNOWN = "unknown"
    }
}
