package com.micronaut.bug.trace

import org.slf4j.MDC
import org.springframework.core.task.TaskDecorator

/**
 * Декоратор для задач [org.springframework.core.task.TaskExecutor] и [org.springframework.scheduling.TaskScheduler],
 * обеспечивающий перенос или инициализацию контекста трассировки между потоками.
 *
 * В Spring Framework выполнение задач с аннотациями `@Async` или `@Scheduled` происходит в отдельных
 * потоках пула, где стандартные ThreadLocal переменные родителя (включая MDC и стек спанов) отсутствуют.
 * Данный декоратор решает эту проблему, управляя жизненным циклом [com.micronaut.bug.config.trace.TraceContext].
 *
 * ### Сценарии использования:
 * 1. **Асинхронные задачи (@Async):**
 *    Декоратор делает "снимок" состояния родительского потока и восстанавливает его в дочернем.
 *    В Grafana Tempo это отображается как дочерний спан (Child Span), связанный с основным запросом.
 *
 * 2. **Планировщик (@Scheduled):**
 *    Поскольку у фоновых задач нет родительского HTTP-запроса, декоратор обнаруживает пустой контекст
 *    и автоматически инициирует новый корневой трейс (Root Trace). Это позволяет мониторить
 *    производительность шедулеров и искать их логи по уникальному `rqId`.
 *
 * ### Механика работы:
 * 1. В момент постановки задачи в очередь выполняется [decorate] (в родительском потоке): создается снимок стека.
 * 2. Перед выполнением задачи (в дочернем потоке):
 *    - Восстанавливается снимок или создается новый корневой контекст.
 *    - Открывается новый спан через `tracer.startSpan`.
 *    - В блоке `finally` спан закрывается, а поток полностью очищается для предотвращения "утечки"
 *      данных между задачами в пуле.
 *
 * ### Пример настройки для @Async и @Scheduled:
 * ```kotlin
 * // Для @Async
 * @Bean
 * fun taskExecutor(tracer: NanoTracer, props: TraceProperties): Executor =
 *     ThreadPoolTaskExecutor().apply {
 *         setTaskDecorator(TracingTaskDecorator(tracer, props))
 *         initialize()
 *     }
 *
 * // Для @Scheduled
 * @Bean
 * fun taskScheduler(tracer: NanoTracer, props: TraceProperties): TaskScheduler =
 *     ThreadPoolTaskScheduler().apply {
 *         setTaskDecorator(TracingTaskDecorator(tracer, props))
 *         initialize()
 *     }
 * ```
 *
 * ### Пример логирования в @Scheduled:
 * ```kotlin
 * @Scheduled(fixedRate = 10000)
 * fun heartBeat() {
 *     // В каждой итерации будет новый rqId в консоли и новый трейс в Tempo
 *     log.info { "System heart beat" }
 * }
 * ```
 *
 * @property tracer Экземпляр [NanoTracer] для управления стеком контекстов.
 */

class TracingTaskDecorator(
    private val tracer: NanoTracer? = null,
) : TaskDecorator {

    override fun decorate(runnable: Runnable): Runnable {
        // Если трейсер не инициализирован, возвращаем задачу как есть
        if (tracer == null) {
            return runnable
        }

        // 1. ЗАХВАТ КОНТЕКСТА В РОДИТЕЛЬСКОМ ПОТОКЕ (Zero-Allocation)
        // Просто копируем 64-битную ссылку на текущую ноду иммутабельного стека
        val parentSpan = tracer.currentSpan()
        val parentMdc = MDC.getCopyOfContextMap()

        return Runnable {
            // 2. ИНИЦИАЛИЗАЦИЯ В ФОНОВОМ ПОТОКЕ ПУЛА
            // Устанавливаем родительский контекст в ThreadLocal фонового потока
            tracer.setSpan(parentSpan)

            // Если в MDC родителя были кастомные бизнес-метрики/ключи, восстанавливаем их
            if (parentMdc != null) {
                MDC.setContextMap(parentMdc)
            }

            // На основе восстановленного контекста открываем изолированный асинхронный спан.
            // Имя класса runnable берется безопасно.
            val asyncSpan = tracer.startSpan("ASYNC: ${runnable.javaClass.simpleName}")

            try {
                runnable.run()
            } catch (e: Exception) {
                // Если фоновая задача упала — фиксируем ошибку в спане
                asyncSpan.error = e
                throw e
            } finally {
                // 3. ОБЯЗАТЕЛЬНАЯ И СТРОГАЯ ОЧИСТКА ПОТОКА ПУЛА
                // Сначала закрываем наш спан (NanoTracer сам откатит поток на parentCtx)
                tracer.stop(asyncSpan)

                // Полностью вычищаем ThreadLocal и MDC перед возвратом потока в пул Spring/Tomcat.
                // Это гарантирует 100% изоляцию данных между независимыми задачами.
                tracer.clearThreadContext()
            }
        }
    }
}
