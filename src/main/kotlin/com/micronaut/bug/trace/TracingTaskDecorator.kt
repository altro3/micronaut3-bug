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
        // Берем только плоские иммутабельные данные (строки), никаких ссылок на объекты спанов!
        val parent = tracer.currentSpan()
        val traceId = parent?.traceId
        val spanId = parent?.spanId
        val sampled = parent?.sampled
        val baggage = parent?.baggage
        val propagationHeaders = parent?.propagationHeaders
        val traceState = parent?.traceState

        // Захватываем состояние MDC родительского потока
        val parentMdc = MDC.getCopyOfContextMap()

        return Runnable {
            // 2. ЗАПОМИНАЕМ СТАРЫЙ КОНТЕКСТ ФОНОВОГО ПОТОКА (для корректного восстановления)
            val originalSpan = tracer.currentSpan()
            val originalMdc = MDC.getCopyOfContextMap()

            // Инициализируем MDC родительскими данными для логов внутри runnable
            if (parentMdc != null) {
                MDC.setContextMap(parentMdc)
            } else {
                MDC.clear()
            }

            // Стартуем изолированный спан для асинхронной задачи.
            // Передаем ID родителя. Теперь фоновый поток имеет свою независимую ветку и не мутирует родительский спан!
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
                // 3. СИММЕТРИЧНОЕ ВОССТАНОВЛЕНИЕ ПОТОКА ПУЛА
                // Закрываем асинхронный спан
                tracer.stop(asyncSpan)

                // Вместо разрушительного clearThreadSpan(), который зачищает всё в ноль,
                // мы возвращаем фоновый поток строго в то состояние, в котором он был до задачи.
                tracer.setSpan(originalSpan)

                if (originalMdc != null) {
                    MDC.setContextMap(originalMdc)
                } else {
                    MDC.clear()
                }
            }
        }
    }
}
