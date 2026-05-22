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
        val oldMdc = MDC.getCopyOfContextMap()

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

        return TraceStateSnapshot(oldSpan, oldMdc)
    }

    override fun restoreThreadContext(context: CoroutineContext, oldState: TraceStateSnapshot) {
        tracer.setSpanInternal(oldState.oldSpan)
        if (oldState.oldMdc != null) {
            MDC.setContextMap(oldState.oldMdc)
        } else {
            MDC.remove(MDC_TRACE_ID)
            MDC.remove(MDC_SPAN_ID)
            MDC.remove(MDC_TRACE_FLAGS)
        }
    }

    class TraceStateSnapshot(
        val oldSpan: NanoSpan?,
        val oldMdc: Map<String, String>?
    )
}
