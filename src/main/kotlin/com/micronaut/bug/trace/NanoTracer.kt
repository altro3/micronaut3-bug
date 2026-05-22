package com.micronaut.bug.trace

import com.micronaut.bug.trace.NanoTraceFilter.Companion.HEADER_API_KEY
import com.micronaut.bug.trace.config.TraceProperties
import com.micronaut.bug.trace.otlp.TraceBatcher
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import kotlinx.coroutines.withContext
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.CONTENT_LENGTH
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpHeaders.HOST
import org.springframework.http.HttpHeaders.USER_AGENT
import org.springframework.http.HttpMethod
import java.time.Instant
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.ThreadLocalRandom

class NanoTracer(
    @PublishedApi
    internal val batcher: TraceBatcher,
    private val traceProps: TraceProperties,
) {
    private val threadContext = ThreadLocal<NanoSpan?>()

    val clockOffsetNanos: Long = (Instant.now().let { it.epochSecond * NANOS_PER_SECOND + it.nano }) - System.nanoTime()

    @PublishedApi
    internal val reportPool = ConcurrentLinkedQueue<TraceReport>()

    @PublishedApi
    internal fun getCurrentEpochNanos(): Long = System.nanoTime() + clockOffsetNanos

    fun currentSpan(): NanoSpan? = threadContext.get()

    internal fun setSpan(ctx: NanoSpan?) {
        threadContext.set(ctx)
        syncMdc(ctx)
    }

    private fun pushSpan(ctx: NanoSpan): NanoSpan {
        threadContext.set(ctx)
        syncMdc(ctx)
        return ctx
    }

    @PublishedApi
    internal fun syncMdc(current: NanoSpan?) {
        if (current != null) {
            MDC.put(MDC_TRACE_ID, current.traceId)
            MDC.put(MDC_SPAN_ID, current.spanId)
            MDC.put(MDC_TRACE_FLAGS, if (current.sampled) TRACE_FLAG_SAMPLED else TRACE_FLAG_NOT_SAMPLED)
        } else {
            MDC.remove(MDC_TRACE_ID)
            MDC.remove(MDC_SPAN_ID)
            MDC.remove(MDC_TRACE_FLAGS)
        }
    }

    fun startTrace(
        name: String, remoteTraceId: String? = null, remoteParentId: String? = null,
        sampled: Boolean? = null, baggage: Map<String, String>? = null,
        propagationHeaders: Map<String, String>? = null, traceState: String? = null,
    ): NanoSpan {
        val effectiveSampled = sampled ?: (remoteParentId != null || ThreadLocalRandom.current().nextDouble() < traceProps.sampleRate)

        return pushSpan(
            NanoSpan(
                traceId = remoteTraceId ?: TraceIdGenerator.generate(),
                spanId = TraceIdGenerator.generateSpanId(),
                parentId = remoteParentId,
                name = name,
                startEpochNanos = getCurrentEpochNanos(),
                sampled = effectiveSampled,
                baggage = baggage,
                propagationHeaders = propagationHeaders,
                traceState = traceState,
                parentContext = threadContext.get(),
            )
        )
    }

    fun startSpan(name: String, parent: NanoSpan? = null): NanoSpan {
        val effectiveParent = parent ?: currentSpan() ?: return startTrace(name)

        return pushSpan(
            NanoSpan(
                traceId = effectiveParent.traceId,
                spanId = TraceIdGenerator.generateSpanId(),
                parentId = effectiveParent.spanId,
                name = name,
                startEpochNanos = getCurrentEpochNanos(),
                sampled = effectiveParent.sampled,
                baggage = effectiveParent.baggage,
                propagationHeaders = effectiveParent.propagationHeaders,
                traceState = effectiveParent.traceState,
                parentContext = effectiveParent,
            )
        )
    }

    fun stop(
        ctx: NanoSpan,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
        attrs: Map<String, Any>? = null,
        forceExport: Boolean = false,
    ) {
        if (!ctx.tryClose()) return

        val current = threadContext.get()

        if (current === ctx) {
            setSpan(ctx.parentContext)
        } else if (current != null) {
            val newChain = removeFromChain(current, ctx)
            setSpan(newChain)
        }

        val isError = ctx.error != null || status == StatusCode.STATUS_CODE_ERROR
        val shouldExport = isError || forceExport || ctx.sampled

        if (!shouldExport) return

        batcher.enqueue(
            traceIdHex = ctx.traceId,
            spanIdHex = ctx.spanId,
            parentIdHex = ctx.parentId,
            name = ctx.name,
            startEpochNanos = ctx.startEpochNanos,
            endEpochNanos = getCurrentEpochNanos(),
            status = status,
            kind = kind,
            userAttrs = attrs,
            baggage = ctx.baggage,
            propagationHeaders = ctx.propagationHeaders,
            error = ctx.error,
        )
    }


    private fun removeFromChain(node: NanoSpan, targetToRemove: NanoSpan): NanoSpan? {
        if (node === targetToRemove) {
            return node.parentContext
        }

        val parent = node.parentContext ?: return node

        val newParent = removeFromChain(parent, targetToRemove)

        if (newParent === parent) {
            return node
        }

        return NanoSpan(
            traceId = node.traceId,
            spanId = node.spanId,
            parentId = node.parentId,
            name = node.name,
            startEpochNanos = node.startEpochNanos,
            sampled = node.sampled,
            baggage = node.baggage,
            propagationHeaders = node.propagationHeaders,
            traceState = node.traceState,
            parentContext = newParent,
        )
    }

    suspend inline fun <T> trace(
        name: String,
        kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
        crossinline block: suspend (TraceReport) -> T,
    ): T {
        val ctx = startSpan(name)
        // Беру репорт из пула — ноль аллокаций HashMap!
        val report = reportPool.poll() ?: TraceReport()

        try {
            // Легальный и безопасный перенос контекста через встроенный корутинный механизм
            return withContext(TraceElement(ctx, this)) {
                block(report)
            }
        } catch (e: Exception) {
            ctx.error = e
            report.status = StatusCode.STATUS_CODE_ERROR
            throw e
        } finally {
            stop(ctx = ctx, status = report.status, kind = kind, attrs = report.attrs)
            report.clear()
            reportPool.offer(report)
        }
    }

    fun getTraceParent(): String? {
        val current = currentSpan() ?: return null
        val sb = tlStringBuilder.get()
        sb.setLength(0)
        sb.append(TRACEPARENT_PREFIX)
            .append(current.traceId)
            .append('-')
            .append(current.spanId)
            .append('-')
            .append(if (current.sampled) TRACE_FLAG_SAMPLED else TRACE_FLAG_NOT_SAMPLED)
        return sb.toString()
    }

    fun clearThreadContext() {
        threadContext.remove()
        MDC.remove(MDC_TRACE_ID)
        MDC.remove(MDC_SPAN_ID)
        MDC.remove(MDC_TRACE_FLAGS)
    }

    companion object {
        const val MDC_TRACE_ID = "traceId"
        const val MDC_TRACE_FLAGS = "traceFlags"
        const val MDC_SPAN_ID = "spanId"
        const val MDC_SOURCE = "source"
        const val MDC_TARGET = "target"
        const val MDC_SUB_TITLE = "subTitle"
        const val MDC_MAIN_STAT = "mainstat"
        const val MDC_COLOR = "color"

        const val HEADER_X_SENDER = "x-sender"
        const val HEADER_TRACEPARENT = "traceparent"
        const val HEADER_BAGGAGE = "baggage"
        const val HEADER_TRACESTATE = "tracestate"
        const val TRACEPARENT_PREFIX = "00-"

        const val TRACE_FLAG_SAMPLED = "01"
        const val TRACE_FLAG_NOT_SAMPLED = "00"

        val ATTR_ERROR_MESSAGE = "error.message"

        @JvmField
        val OTEL_MAPPED_HEADERS = setOf(
            CONTENT_LENGTH.lowercase(),
            CONTENT_TYPE.lowercase(),
            USER_AGENT.lowercase(),
            HOST.lowercase(),
            HEADER_TRACEPARENT.lowercase(),
            HEADER_TRACESTATE.lowercase(),
            HEADER_BAGGAGE.lowercase(),
        )

        private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(64) }

        const val ATTR_SERVICE_NAME = "service.name"
        const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"
        const val ATTR_SERVER_ADDRESS = "server.address"
        const val ATTR_SERVER_PORT = "server.port"
        const val ATTR_CLIENT_ADDRESS = "client.address"
        const val ATTR_HTTP_REQUEST_METHOD = "http.request.method"
        const val ATTR_HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
        const val PREFIX_HTTP_REQUEST_HEADER = "http.request.header."
        const val ATTR_URL_FULL = "url.full"
        const val ATTR_URL_SCHEME = "url.scheme"
        const val ATTR_URL_PATH = "url.path"
        const val ATTR_URL_QUERY = "url.query"
        const val ATTR_USER_AGENT_ORIGINAL = "user_agent.original"
        const val ATTR_HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
        const val ATTR_HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"
        const val PREFIX_HTTP_RESPONSE_HEADER = "http.response.header."
        const val ATTR_EXCEPTION_TYPE = "exception.type"
        const val ATTR_EXCEPTION_MESSAGE = "exception.message"
        const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"
        const val ATTR_CLIENT = "client"
        const val ATTR_SERVER = "server"
        const val ATTR_PEER_SERVICE = "peer.service"
        const val ATTR_ERROR_TYPE = "error.type"
        const val PREFIX_BAGGAGE = "baggage."
        const val PREFIX_PROPAGATION = "prop."
        const val ATTR_HTTP_SLOW_REQUEST = "http.slow_request"

        const val NANOS_PER_SECOND = 1_000_000_000L

        @JvmField
        val METHODS_WITHOUT_BODY = setOf(
            HttpMethod.GET.name(),
            HttpMethod.HEAD.name(),
            HttpMethod.OPTIONS.name(),
            HttpMethod.DELETE.name(),
            HttpMethod.TRACE.name(),
        )

        @JvmField
        val SENSITIVE_HEADERS = setOf(
            HttpHeaders.AUTHORIZATION.lowercase(),
            HttpHeaders.COOKIE.lowercase(),
            HttpHeaders.SET_COOKIE.lowercase(),
            HEADER_API_KEY,
        )

        const val MASK = "***"

        @JvmField
        val MASKED_VALUES = listOf(MASK)

        @JvmStatic
        fun parseBaggage(header: String?): Map<String, String>? {
            if (header.isNullOrBlank()) return null
            val map = HashMap<String, String>()
            var start = 0
            val len = header.length
            while (start < len) {
                var end = header.indexOf(',', start)
                if (end == -1) end = len
                if (start >= end) {
                    start = end + 1; continue
                }
                val equalsIdx = header.indexOf('=', start)
                if (equalsIdx != -1 && equalsIdx < end) {
                    val semicolonIdx = header.indexOf(';', start)
                    val valueEnd = if (semicolonIdx != -1 && semicolonIdx < end) semicolonIdx else end
                    val key = header.substring(start, equalsIdx).trim().lowercase()
                    val value = header.substring(equalsIdx + 1, valueEnd).trim()
                    if (key.isNotEmpty()) map[key] = value
                }
                start = end + 1
            }
            return map
        }

        @JvmStatic
        fun formatBaggage(baggage: Map<String, String>?): String? {
            if (baggage.isNullOrEmpty()) return null

            val sb = StringBuilder(128)
            var isFirst = true

            for ((key, value) in baggage) {
                if (!isFirst) {
                    sb.append(',')
                }
                sb.append(key).append('=').append(value)
                isFirst = false
            }

            return sb.toString()
        }
    }
}
