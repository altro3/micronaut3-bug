package com.micronaut.bug.trace.reactor

import com.micronaut.bug.trace.NanoSpan
import com.micronaut.bug.trace.TraceUtil.MDC_SPAN_ID
import com.micronaut.bug.trace.TraceUtil.MDC_TRACE_ID
import org.slf4j.MDC
import reactor.core.CoreSubscriber
import reactor.core.publisher.Hooks
import reactor.core.publisher.Operators
import reactor.util.context.Context

object ReactorMdcHook {

    private const val HOOK_KEY = "nano-trace-mdc-hook"

    fun init() {
        Hooks.onEachOperator(HOOK_KEY, Operators.lift { _, subscriber ->
            MdcSubscriber(subscriber)
        })
    }

    class MdcSubscriber<T>(private val delegate: CoreSubscriber<T>) : CoreSubscriber<T> {
        override fun currentContext(): Context = delegate.currentContext()

        override fun onSubscribe(s: org.reactivestreams.Subscription) = delegate.onSubscribe(s)

        override fun onNext(t: T) {
            updateMdc()
            delegate.onNext(t)
        }

        override fun onError(t: Throwable) {
            updateMdc()
            delegate.onError(t)
        }

        override fun onComplete() {
            updateMdc()
            delegate.onComplete()
        }

        private fun updateMdc() {
            val ctx = delegate.currentContext()
                .getOrEmpty<NanoSpan>(NanoSpan::class.java)
                .orElse(null)
            if (ctx != null) {
                MDC.put(MDC_TRACE_ID, ctx.traceId)
                MDC.put(MDC_SPAN_ID, ctx.spanId)
            }
        }
    }
}
