package com.micronaut.bug.trace

import org.slf4j.MDC
import org.springframework.core.task.TaskDecorator

/**
 * Декоратор для задач [org.springframework.core.task.TaskExecutor], обеспечивающий
 * перенос контекста трассировки между потоками.
 *
 * В Spring Framework выполнение задач с аннотацией `@Async` или через пул потоков
 * происходит в отдельном потоке, где ThreadLocal переменные родителя (включая MDC и стек спанов)
 * отсутствуют. Данный декоратор делает "снимок" состояния NanoTracer и MDC в момент
 * постановки задачи в очередь и восстанавливает их непосредственно перед выполнением в целевом потоке.
 *
 * ### Механика работы:
 * 1. В родительском потоке выполняется [decorate]: создаются копии стека и MDC.
 * 2. В дочернем потоке (внутри созданного Runnable):
 *    - Сохраняется старое состояние потока (на случай переиспользования в пуле).
 *    - Устанавливается скопированный контекст.
 *    - Выполняется бизнес-логика.
 *    - **Важно:** В блоке `finally` происходит полная очистка контекста для предотвращения "утечки"
 *      данных в следующие задачи этого же потока.
 *
 * ### Пример настройки в конфигурации:
 * ```kotlin
 * @Bean
 * fun taskExecutor(tracer: NanoTracer): Executor {
 *     val executor = ThreadPoolTaskExecutor()
 *     executor.corePoolSize = 5
 *     executor.setTaskDecorator(TracingTaskDecorator(tracer)) // Регистрация декоратора
 *     executor.initialize()
 *     return executor
 * }
 * ```
 *
 * ### Пример использования:
 * ```kotlin
 * @Async
 * fun processInBackground() {
 *     // Здесь MDC.get("rqId") вернет тот же ID, что был в контроллере
 *     log.info { "Задание выполняется в другом потоке, но с тем же traceId" }
 * }
 * ```
 *
 * @property tracer Экземпляр [NanoTracer], содержащий ThreadLocal стек контекстов.
 */
class TracingTaskDecorator(
    private val tracer: NanoTracer? = null,
) : TaskDecorator {

    override fun decorate(runnable: Runnable): Runnable {
        // Если трейсер не прилетел или трейсинг выключен — возвращаем как есть
        if (tracer == null) {
            return runnable
        }
        // 1. Делаем снимок текущего стека NanoTracer и MDC в РОДИТЕЛЬСКОМ потоке
        val parentTraceElement = tracer.dispatcher()
        val mdcSnapshot = MDC.getCopyOfContextMap()

        return Runnable {

            val oldContext = tracer.internalStack.get()
            val oldMdc = MDC.getCopyOfContextMap()

            // 2. Восстанавливаем родительский контекст в новом потоке
            tracer.internalStack.set(parentTraceElement.snapshot)
            // 3. СРАЗУ открываем новый дочерний спан для асинхронной задачи
            // Это создаст новую связь в дереве Tempo
            val asyncSpan = tracer.startSpan("ASYNC: ${runnable.javaClass.simpleName}")

            try {
                // Синхронизируем MDC на основе восстановленного стека
                tracer.syncMdc()
                // Если в MDC были кастомные ключи, которых нет в стеке — докидываем их
                mdcSnapshot?.let { MDC.setContextMap(it) }

                runnable.run()
            } catch (e: Exception) {
                // Если задача упала, фиксируем ошибку в этом специфичном спане
                asyncSpan.error = e
                throw e
            } finally {
                // 3. Очищаем или восстанавливаем старое состояние (вежливость к пулу потоков)
                if (oldContext != null) {
                    tracer.internalStack.set(oldContext)
                } else {
                    tracer.internalStack.remove()
                }
                if (oldMdc != null) {
                    MDC.setContextMap(oldMdc)
                } else {
                    MDC.clear()
                }
            }
        }
    }
}
