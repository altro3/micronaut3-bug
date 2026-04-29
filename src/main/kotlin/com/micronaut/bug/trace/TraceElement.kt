package com.micronaut.bug.trace

import kotlinx.coroutines.ThreadContextElement
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

/**
 * Элемент контекста корутины, обеспечивающий перенос стека трейсинга между потоками.
 * Реализует [ThreadContextElement] для синхронизации [ThreadLocal] хранилища и MDC.
 */
class TraceElement(
    private val tracer: NanoTracer,
    private val snapshot: ArrayDeque<TraceContext>
) : ThreadContextElement<ArrayDeque<TraceContext>?>, AbstractCoroutineContextElement(Key) {

    /**
     * Ключ для поиска элемента в CoroutineContext.
     */
    companion object Key : CoroutineContext.Key<TraceElement>

    /**
     * Вызывается перед возобновлением корутины на текущем потоке.
     * Устанавливает стек корутины в ThreadLocal потока и возвращает предыдущее состояние.
     */
    override fun updateThreadContext(context: CoroutineContext): ArrayDeque<TraceContext>? {
        val oldState = tracer.internalStack.get()

        // Устанавливаем текущий стек корутины в хранилище потока
        tracer.internalStack.set(snapshot)

        // Синхронизируем MDC, чтобы логи этого потока получили актуальные traceId/spanId
        tracer.syncMdc()

        return oldState
    }

    /**
     * Вызывается после того, как корутина была приостановлена (suspended) или завершена.
     * Восстанавливает состояние ThreadLocal и MDC, которое было в потоке до выполнения корутины.
     */
    override fun restoreThreadContext(context: CoroutineContext, oldState: ArrayDeque<TraceContext>?) {
        if (!oldState.isNullOrEmpty()) {
            tracer.internalStack.set(oldState)
        } else {
            // Если стека до этого не было, полностью очищаем ThreadLocal
            tracer.internalStack.remove()
        }

        // Синхронизируем MDC обратно под старое состояние потока
        tracer.syncMdc()
    }
}
