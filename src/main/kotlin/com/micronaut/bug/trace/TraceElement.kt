package com.micronaut.bug.trace

import kotlinx.coroutines.ThreadContextElement
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

class TraceElement(
    private val contextToRestore: NanoSpan?,
    private val tracer: NanoTracer,
) : ThreadContextElement<NanoSpan?>, AbstractCoroutineContextElement(Key) {

    companion object Key : CoroutineContext.Key<TraceElement>

    override fun updateThreadContext(context: CoroutineContext): NanoSpan? {
        val oldContext = tracer.currentSpan()
        tracer.setSpan(contextToRestore)
        return oldContext
    }

    override fun restoreThreadContext(context: CoroutineContext, oldState: NanoSpan?) {
        tracer.setSpan(oldState)
    }
}
