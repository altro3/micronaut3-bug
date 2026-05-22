package com.micronaut.bug.trace

import org.slf4j.MDC
import org.springframework.core.task.TaskDecorator

class TracingTaskDecorator(
    private val tracer: NanoTracer? = null,
) : TaskDecorator {

    override fun decorate(runnable: Runnable): Runnable {
        if (tracer == null) {
            return runnable
        }

        // 1. ЗАХВАТ КОНТЕКСТА В РОДИТЕЛЬСКОМ ПОТОКЕ
        val parent = tracer.currentSpan()
        val traceId = parent?.traceId
        val spanId = parent?.spanId
        val sampled = parent?.sampled
        val baggage = parent?.baggage
        val propagationHeaders = parent?.propagationHeaders
        val traceState = parent?.traceState

        val parentMdc = MDC.getCopyOfContextMap()

        return Runnable {
            val originalSpan = tracer.currentSpan()
            val originalMdc = MDC.getCopyOfContextMap()

            if (parentMdc != null) {
                MDC.setContextMap(parentMdc)
            } else {
                MDC.clear()
            }

            val asyncSpan = tracer.startTrace(
                name = "ASYNC: ${runnable.javaClass.simpleName}",
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
                tracer.setSpanInternal(originalSpan)

                if (originalMdc != null) {
                    MDC.setContextMap(originalMdc)
                } else {
                    MDC.clear()
                }
            }
        }
    }
}
