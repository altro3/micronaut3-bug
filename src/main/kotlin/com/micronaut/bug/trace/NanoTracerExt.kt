package com.micronaut.bug.trace

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext

/**
 * NanoTracerExt.kt
 *
 * Набор расширений для [CoroutineScope], обеспечивающих бесшовную интеграцию
 * корутин с системой распределенной трассировки [NanoTracer].
 */

/**
 * Запускает новую корутину с сохранением текущего контекста трассировки.
 *
 * Метод делает снимок (snapshot) текущего стека спанов и MDC из родительского потока
 * и восстанавливает его внутри корутины. Это гарантирует, что все логи внутри
 * асинхронного блока будут содержать тот же `rqId`.
 *
 * ### Example:
 * ```
 * scope.launchTraced(tracer) {
 *     log.info { "Фоновая задача с сохранением traceId" }
 * }
 * ```
 *
 * @param tracer экземпляр активного трейсера.
 * @param context дополнительные элементы контекста корутины (например, Dispatchers.IO).
 * @param block приостанавливаемый блок кода для выполнения.
 * @return [Job] запущенной корутины.
 */
fun CoroutineScope.launchTraced(
    tracer: NanoTracer,
    context: CoroutineContext = EmptyCoroutineContext,
    block: suspend CoroutineScope.() -> Unit
): Job {
    return launch(context + tracer.dispatcher()) {
        block()
    }
}

/**
 * Запускает корутину c возвращаемым результатом и пробросом контекста NanoTracer.
 *
 * Аналог [launchTraced], но возвращает [Deferred], что позволяет дождаться
 * результата выполнения асинхронной операции.
 *
 * ### Example:
 * ```
 * val result = scope.asyncTraced(tracer) {
 *     fetchData()
 * }.await()
 * ```
 *
 * @param T тип возвращаемого значения.
 * @param tracer экземпляр активного трейсера.
 * @param context дополнительные элементы контекста корутины.
 * @return [Deferred] с результатом вычисления.
 */
fun <T> CoroutineScope.asyncTraced(
    tracer: NanoTracer,
    context: CoroutineContext = EmptyCoroutineContext,
    block: suspend CoroutineScope.() -> T
): Deferred<T> {
    return async(context + tracer.dispatcher()) {
        block()
    }
}

/**
 * Запускает корутину и автоматически оборачивает её выполнение в новый именованный спан.
 *
 * Самый мощный метод для фоновых задач:
 * 1. Пробрасывает контекст родителя (traceId).
 * 2. Создает вложенный спан в Tempo.
 * 3. Автоматически фиксирует ошибки (stacktrace) в этом спане.
 *
 * ### Example:
 * ```
 * scope.launchNewSpan(tracer, "process-upload") { report ->
 *     val size = uploadFile()
 *     report.attrs["file.size"] = size
 *     log.info { "Загрузка завершена" }
 * }
 * ```
 *
 * @param tracer экземпляр активного трейсера.
 * @param spanName имя для нового спана в Grafana Tempo.
 * @param context дополнительные элементы контекста корутины.
 * @param block приостанавливаемый блок кода, принимающий [TraceReport] для записи атрибутов.
 * @return [Job] запущенной корутины.
 */
fun CoroutineScope.launchNewSpan(
    tracer: NanoTracer,
    spanName: String,
    context: CoroutineContext = EmptyCoroutineContext,
    block: suspend (NanoTracer.TraceReport) -> Unit
): Job {
    return launch(context + tracer.dispatcher()) {
        tracer.trace(spanName) { report ->
            block(report)
        }
    }
}