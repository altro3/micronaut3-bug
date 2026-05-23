package com.micronaut.bug.trace

import com.micronaut.bug.trace.TraceElement.TraceStateSnapshot
import com.micronaut.bug.trace.TraceUtil.MDC_SPAN_ID
import com.micronaut.bug.trace.TraceUtil.MDC_TRACE_FLAGS
import com.micronaut.bug.trace.TraceUtil.MDC_TRACE_ID
import com.micronaut.bug.trace.TraceUtil.TRACE_FLAG_NOT_SAMPLED
import com.micronaut.bug.trace.TraceUtil.TRACE_FLAG_SAMPLED
import kotlinx.coroutines.ThreadContextElement
import org.slf4j.MDC
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

class TraceElement(
    private val spanToRestore: NanoSpan?,
    private val tracer: NanoTracer,
) : ThreadContextElement<TraceStateSnapshot>, AbstractCoroutineContextElement(Key) {

    companion object Key : CoroutineContext.Key<TraceElement>

    override fun updateThreadContext(context: CoroutineContext): TraceStateSnapshot {
        val oldSpan = tracer.currentSpan()

        val oldTraceId = MDC.get(MDC_TRACE_ID)
        val oldSpanId = MDC.get(MDC_SPAN_ID)
        val oldTraceFlags = MDC.get(MDC_TRACE_FLAGS)

        tracer.setSpanInternal(spanToRestore)

        if (spanToRestore != null) {
            MDC.put(MDC_TRACE_ID, spanToRestore.traceId)
            MDC.put(MDC_SPAN_ID, spanToRestore.spanId)
            MDC.put(MDC_TRACE_FLAGS, if (spanToRestore.sampled) TRACE_FLAG_SAMPLED else TRACE_FLAG_NOT_SAMPLED)
        } else {
            MDC.remove(MDC_TRACE_ID)
            MDC.remove(MDC_SPAN_ID)
            MDC.remove(MDC_TRACE_FLAGS)
        }

        return TraceStateSnapshot(oldSpan, oldTraceId, oldSpanId, oldTraceFlags)
    }

    override fun restoreThreadContext(context: CoroutineContext, oldState: TraceStateSnapshot) {
        tracer.setSpanInternal(oldState.oldSpan)

        if (oldState.oldTraceId != null) {
            MDC.put(MDC_TRACE_ID, oldState.oldTraceId)
        } else {
            MDC.remove(MDC_TRACE_ID)
        }
        if (oldState.oldSpanId != null) {
            MDC.put(MDC_SPAN_ID, oldState.oldSpanId)
        } else {
            MDC.remove(MDC_SPAN_ID)
        }
        if (oldState.oldTraceFlags != null) {
            MDC.put(MDC_TRACE_FLAGS, oldState.oldTraceFlags)
        } else {
            MDC.remove(MDC_TRACE_FLAGS)
        }
    }

    class TraceStateSnapshot(
        val oldSpan: NanoSpan?,
        val oldTraceId: String?,
        val oldSpanId: String?,
        val oldTraceFlags: String?
    )
}
