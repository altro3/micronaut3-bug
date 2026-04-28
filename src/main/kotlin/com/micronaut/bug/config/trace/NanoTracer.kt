package com.micronaut.bug.config.trace

import com.micronaut.bug.config.trace.TraceIdGenerator.generate
import com.micronaut.bug.config.trace.TraceIdGenerator.generateSpanId
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import kotlinx.coroutines.withContext
import org.slf4j.MDC
import java.time.Instant
import java.util.Deque

class NanoTracer(
    @PublishedApi
    internal val exporter: TempoExporter,
) {
    @PublishedApi
    internal val internalStack = ThreadLocal<Deque<TraceContext>>.withInitial { ArrayDeque<TraceContext>() }

    private val clockOffsetNanos: Long by lazy {
        val now = Instant.now()
        (now.epochSecond * 1_000_000_000L + now.nano) - System.nanoTime()
    }

    @PublishedApi
    internal fun getCurrentEpochNanos(): Long = System.nanoTime() + clockOffsetNanos

    /**
     * Стартует новый трейс (корневой спан). Генерирует новый traceId.
     */
    fun startTrace(name: String): TraceContext {
        val ctx = TraceContext(
            traceId = generate(), // Новый трейс
            spanId = generateSpanId(),
            parentId = null,
            name = name,
            startEpochNanos = getCurrentEpochNanos()
        )
        return pushAndSync(ctx)
    }

    /**
     * Стартует вложенный спан. Если в стеке ничего нет, ведет себя как startTrace.
     */
    fun startSpan(name: String): TraceContext {
        val stack = internalStack.get()
        val parent = stack.firstOrNull()

        if (parent == null) {
            return startTrace(name)
        }

        val ctx = TraceContext(
            traceId = parent.traceId, // Наследуем traceId
            spanId = generateSpanId(),
            parentId = parent.spanId,
            name = name,
            startEpochNanos = getCurrentEpochNanos()
        )
        return pushAndSync(ctx)
    }

    /**
     * Завершает операцию и отправляет данные в экспортер.
     */
    fun stop(
        ctx: TraceContext,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        attrs: Map<String, Any> = emptyMap()
    ) {
        val endNs = getCurrentEpochNanos()
        val stack = internalStack.get()

        // Удаляем именно этот контекст (даже если закрыли не в том порядке)
        stack.remove(ctx)

        exporter.enqueue(
            traceIdHex = ctx.traceId,
            spanIdHex = ctx.spanId,
            parentIdHex = ctx.parentId,
            name = ctx.name,
            startEpochNanos = ctx.startEpochNanos,
            endEpochNanos = endNs,
            status = status,
            attrs = attrs
        )
        syncMdc()
    }

    /**
     * Удобная обертка для автоматического трейсинга корутин.
     * Обрабатывает исключения и автоматически закрывает спан.
     */
    suspend inline fun <T> trace(
        name: String,
        crossinline block: suspend (TraceReport) -> T
    ): T {
        val ctx = startSpan(name)
        val report = TraceReport()

        try {
            // Передаем ссылку на stack напрямую в TraceElement
            return withContext(TraceElement(this, internalStack.get())) {
                block(report)
            }
        } catch (e: Exception) {
            report.status = StatusCode.STATUS_CODE_ERROR
            report.attrs["error.message"] = e.message ?: e.javaClass.simpleName
            report.attrs["error.type"] = e.javaClass.name
            throw e
        } finally {
            stop(ctx, report.status, report.attrs)
        }
    }

    private fun pushAndSync(ctx: TraceContext): TraceContext {
        internalStack.get().addFirst(ctx)
        syncMdc()
        return ctx
    }

    /**
     * Синхронизирует MDC текущего потока с головой стека.
     */
    @PublishedApi
    internal fun syncMdc() {
        val current = internalStack.get().firstOrNull()
        if (current != null) {
            MDC.put("traceId", current.traceId)
            MDC.put("spanId", current.spanId)
        } else {
            MDC.remove("traceId")
            MDC.remove("spanId")
        }
    }

    /**
     * Вспомогательный класс для сбора результатов внутри блока [NanoTracer.trace].
     * Использует значения по умолчанию, чтобы пользователь мог не указывать их при старте.
     */
    class TraceReport(
        var status: StatusCode = StatusCode.STATUS_CODE_OK,
        val attrs: MutableMap<String, Any> = mutableMapOf()
    ) {
        /**
         * Удобный метод для быстрой пометки спана как ошибочного.
         */
        fun error(message: String?) {
            status = StatusCode.STATUS_CODE_ERROR
            message?.let { attrs["error.message"] = it }
        }
    }
}
