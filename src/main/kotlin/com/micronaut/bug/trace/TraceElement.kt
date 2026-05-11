package com.micronaut.bug.trace

import kotlinx.coroutines.ThreadContextElement
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

class TraceElement(
    @PublishedApi
    internal val snapshot: ArrayDeque<TraceContext>,
    private val tracer: NanoTracer,
) : ThreadContextElement<ArrayDeque<TraceContext>?>, AbstractCoroutineContextElement(Key) {

    companion object Key : CoroutineContext.Key<TraceElement>

    override fun updateThreadContext(context: CoroutineContext): ArrayDeque<TraceContext>? {
        val oldState = tracer.internalStack.get()

        tracer.internalStack.set(snapshot)

        tracer.syncMdc()

        return oldState
    }

    override fun restoreThreadContext(context: CoroutineContext, oldState: ArrayDeque<TraceContext>?) {
        if (!oldState.isNullOrEmpty()) {
            tracer.internalStack.set(oldState)
        } else {
            tracer.internalStack.remove()
        }

        tracer.syncMdc()
    }
}
