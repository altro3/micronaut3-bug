package com.micronaut.bug.trace

import com.micronaut.bug.trace.TraceUtil.MDC_SPAN_ID
import com.micronaut.bug.trace.TraceUtil.MDC_TRACE_FLAGS
import com.micronaut.bug.trace.TraceUtil.MDC_TRACE_ID
import com.micronaut.bug.trace.TraceUtil.NANOS_PER_SECOND
import com.micronaut.bug.trace.TraceUtil.TRACEPARENT_PREFIX
import com.micronaut.bug.trace.TraceUtil.TRACE_FLAG_NOT_SAMPLED
import com.micronaut.bug.trace.TraceUtil.TRACE_FLAG_SAMPLED
import com.micronaut.bug.trace.config.TraceProperties
import com.micronaut.bug.trace.otlp.TraceBatcher
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import kotlinx.coroutines.withContext
import org.slf4j.MDC
import java.time.Instant
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.ThreadLocalRandom

class NanoTracer(
    @PublishedApi
    internal val batcher: TraceBatcher,
    private val traceProps: TraceProperties,
) {
    private val threadSpan = ThreadLocal<NanoSpan?>()
    private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(64) }

    val clockOffsetNanos: Long = (Instant.now().let { it.epochSecond * NANOS_PER_SECOND + it.nano }) - System.nanoTime()

    @PublishedApi
    internal val reportPool = ConcurrentLinkedQueue<TraceReport>()

    @PublishedApi
    internal fun getCurrentEpochNanos(): Long = System.nanoTime() + clockOffsetNanos

    fun currentSpan(): NanoSpan? = threadSpan.get()

    fun setSpan(span: NanoSpan?) {
        threadSpan.set(span)
        syncMdc(span)
    }

    internal fun setSpanInternal(span: NanoSpan?) {
        threadSpan.set(span)
    }

    private fun pushSpan(span: NanoSpan): NanoSpan {
        threadSpan.set(span)
        syncMdc(span)
        return span
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
                parentSpan = threadSpan.get(),
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
                parentSpan = effectiveParent,
            )
        )
    }

    fun stop(
        span: NanoSpan,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
        attrs: Map<String, Any>? = null,
        forceExport: Boolean = false,
    ) {
        if (!span.tryClose()) return

        val current = threadSpan.get()

        if (current === span) {
            setSpan(span.parentSpan)
        } else if (current != null) {
            val newChain = removeFromChain(current, span)
            setSpan(newChain)
        }

        val isError = span.error != null || status == StatusCode.STATUS_CODE_ERROR
        val shouldExport = isError || forceExport || span.sampled

        if (!shouldExport) return

        batcher.enqueue(
            traceIdHex = span.traceId,
            spanIdHex = span.spanId,
            parentIdHex = span.parentId,
            name = span.name,
            startEpochNanos = span.startEpochNanos,
            endEpochNanos = getCurrentEpochNanos(),
            status = status,
            kind = kind,
            userAttrs = attrs,
            baggage = span.baggage,
            propagationHeaders = span.propagationHeaders,
            error = span.error,
        )
    }

    private fun removeFromChain(node: NanoSpan, targetToRemove: NanoSpan): NanoSpan? {
        if (node === targetToRemove) {
            return node.parentSpan
        }

        var current = node
        while (current.parentSpan != null) {
            if (current.parentSpan === targetToRemove) {
                current.parentSpan = targetToRemove.parentSpan
                break
            }
            current = current.parentSpan ?: break
        }
        return node
    }

    suspend inline fun <T> trace(
        name: String,
        kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
        crossinline block: suspend (TraceReport) -> T,
    ): T {
        val span = startSpan(name)
        val report = reportPool.poll() ?: TraceReport()

        try {
            return withContext(TraceElement(span, this)) {
                try {
                    block(report)
                } catch (e: Exception) {
                    span.error = e
                    report.status = StatusCode.STATUS_CODE_ERROR
                    throw e
                } finally {
                    stop(span = span, status = report.status, kind = kind, attrs = report.attrs)
                }
            }
        } finally {
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

    fun clearThreadSpan() {
        threadSpan.remove()
        MDC.remove(MDC_TRACE_ID)
        MDC.remove(MDC_SPAN_ID)
        MDC.remove(MDC_TRACE_FLAGS)
    }
}
