package com.micronaut.bug.trace

import org.slf4j.MDC
import org.springframework.core.task.TaskDecorator

class TracingTaskDecorator(
    private val tracer: NanoTracer? = null,
) : TaskDecorator {

    override fun decorate(runnable: Runnable): Runnable {
        if (tracer == null) return runnable

        val parent = tracer.currentSpan()
        val traceId = parent?.traceId
        val spanId = parent?.spanId
        val sampled = parent?.sampled
        val baggage = parent?.baggage
        val propagationHeaders = parent?.propagationHeaders
        val traceState = parent?.traceState

        val parentMdc = MDC.getCopyOfContextMap()

        return Runnable {
            val isScheduled = traceId == null

            val originalMdc = if (!isScheduled) MDC.getCopyOfContextMap() else null

            if (isScheduled) {
                tracer.clearThreadSpan()
                MDC.clear()
            } else {
                if (parentMdc != null) {
                    MDC.setContextMap(parentMdc)
                } else {
                    MDC.clear()
                }
            }

            val spanPrefix = if (isScheduled) PREFIX_SCHEDULED else PREFIX_ASYNC

            val asyncSpan = tracer.startTrace(
                name = "$spanPrefix: ${runnable.javaClass.simpleName}",
                remoteTraceId = traceId,
                remoteParentId = spanId,
                sampled = sampled,
                baggage = baggage,
                propagationHeaders = propagationHeaders,
                traceState = traceState
            )

            try {
                runnable.run()
            } catch (e: Exception) {
                asyncSpan.error = e
                throw e
            } finally {
                tracer.stop(asyncSpan)

                if (isScheduled) {
                    tracer.clearThreadSpan()
                    MDC.clear()
                } else {
                    if (originalMdc != null) {
                        MDC.setContextMap(originalMdc)
                    } else {
                        MDC.clear()
                    }
                }
            }
        }
    }

    companion object {
        private const val PREFIX_SCHEDULED = "SCHEDULED"
        private const val PREFIX_ASYNC = "ASYNC"
    }
}
