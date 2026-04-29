package com.micronaut.bug.trace

import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_ERROR_MESSAGE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.TempoExporter.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_BAGGAGE
import com.micronaut.bug.trace.TempoExporter.Companion.PREFIX_PROPAGATION
import com.micronaut.bug.trace.TraceIdGenerator.generate
import com.micronaut.bug.trace.TraceIdGenerator.generateSpanId
import io.opentelemetry.proto.trace.v1.Span.SpanKind
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
        (now.epochSecond * NANOS_PER_SECOND + now.nano) - System.nanoTime()
    }

    @PublishedApi
    internal fun getCurrentEpochNanos(): Long = System.nanoTime() + clockOffsetNanos

    /**
     * Возвращает текущий активный контекст (верхушка стека).
     */
    fun currentContext(): TraceContext? = internalStack.get().firstOrNull()

    /**
     * Формирует значение заголовка 'traceparent' согласно стандарту W3C Trace Context.
     * Используется для проброса контекста (propagation) в исходящие HTTP-запросы.
     *
     * Формат: 00-{traceId}-{spanId}-{flags}
     * - 00: версия протокола.
     * - traceId: 32-символьный hex-идентификатор всей цепочки.
     * - spanId: 16-символьный hex-идентификатор текущего сегмента (спана).
     * - 01: флаги (в данном случае 'sampled', означающий обязательную запись трейса).
     *
     * @return Строка заголовка или null, если контекст трейсинга отсутствует.
     */
    fun getTraceParent(): String? {
        val current = internalStack.get().firstOrNull() ?: return null
        return "$TRACEPARENT_PREFIX${current.traceId}-${current.spanId}-01"
    }

    /**
     * Формирует и активирует корневой контекст трассировки для текущего потока.
     * Используется в точках входа (например, в [NanoTraceFilter]).
     *
     * @param name Имя операции (например, "POST /api/v1/data").
     * @param remoteTraceId Внешний ID трейса (из заголовка traceparent или X-Request-ID).
     * @param remoteParentId ID родительского спана из внешней системы.
     * @param baggage Набор данных, полученный из стандартного W3C-заголовка "baggage".
     * @param propagationHeaders Заголовки, которые необходимо пробрасывать "как есть" во все клиенты.
     * @return Созданный и помещенный в стек [TraceContext].
     */
    fun startTrace(
        name: String,
        remoteTraceId: String? = null,
        remoteParentId: String? = null,
        baggage: Map<String, String> = emptyMap(),
        propagationHeaders: Map<String, String> = emptyMap(),
    ): TraceContext {
        val ctx = TraceContext(
            traceId = remoteTraceId ?: generate(),
            spanId = generateSpanId(),
            parentId = remoteParentId, // Если он пришел, мы станем вложенным трейсом
            name = name,
            startEpochNanos = getCurrentEpochNanos(),
            baggage = baggage,
            propagationHeaders = propagationHeaders,
        )
        return pushAndSync(ctx)
    }

    /**
     * Создает и активирует дочерний спан, наследуя контекст от текущего активного спана.
     * Если активный контекст отсутствует, метод автоматически создает корневой трейс.
     *
     * Наследует [TraceContext.baggage] и [TraceContext.propagationHeaders], обеспечивая
     * сквозную передачу метаданных по всей цепочке вызовов внутри сервиса и за его пределы.
     *
     * @param name Имя вложенной операции (например, "SERVICE: processOrder" или "CLIENT: callPaymentApi").
     * @return Новый [TraceContext], ставший вершиной стека в текущем потоке.
     */
    fun startSpan(name: String): TraceContext {
        val parent = currentContext()
        if (parent == null) {
            return startTrace(name)
        }

        val ctx = TraceContext(
            traceId = parent.traceId, // Наследуем traceId
            spanId = generateSpanId(),
            parentId = parent.spanId,
            name = name,
            startEpochNanos = getCurrentEpochNanos(),
            baggage = parent.baggage,
            propagationHeaders = parent.propagationHeaders,
        )
        return pushAndSync(ctx)
    }

    /**
     * Завершает операцию, удаляет контекст из стека и отправляет данные в Tempo.
     * Автоматически добавляет [TraceContext.baggage] и [TraceContext.propagationHeaders]
     * в атрибуты спана для обеспечения возможности поиска в UI Grafana.
     */
    fun stop(
        ctx: TraceContext,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        kind: SpanKind = SpanKind.SPAN_KIND_SERVER,
        attrs: Map<String, Any> = emptyMap(),
    ) {
        val endNs = getCurrentEpochNanos()
        val stack = internalStack.get()

        // Удаляем именно этот контекст (даже если закрыли не в том порядке)
        stack.remove(ctx)

        // Объединяем пользовательские атрибуты с метаданными контекста
        val finalAttrs = attrs.toMutableMap().apply {
            // Добавляем багаж (W3C) с префиксом
            ctx.baggage.forEach { (k, v) -> put("$PREFIX_BAGGAGE$k", v) }
            // Добавляем заголовки проброса с префиксом
            ctx.propagationHeaders.forEach { (k, v) -> put("$PREFIX_PROPAGATION$k", v) }
        }

        exporter.enqueue(
            traceIdHex = ctx.traceId,
            spanIdHex = ctx.spanId,
            parentIdHex = ctx.parentId,
            name = ctx.name,
            startEpochNanos = ctx.startEpochNanos,
            endEpochNanos = endNs,
            status = status,
            kind = kind,
            attrs = finalAttrs ,
        )
        syncMdc()
    }

    /**
     * Удобная обертка для автоматического трейсинга корутин.
     * Обрабатывает исключения и автоматически закрывает спан.
     */
    suspend inline fun <T> trace(
        name: String,
        kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
        crossinline block: suspend (TraceReport) -> T,
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
            report.attrs[ATTR_EXCEPTION_TYPE] = e.javaClass.name
            report.attrs[ATTR_EXCEPTION_MESSAGE] = e.message ?: e.javaClass.simpleName
            report.attrs[ATTR_EXCEPTION_STACKTRACE] = e.stackTraceToString()
            throw e
        } finally {
            stop(
                ctx = ctx,
                status = report.status,
                kind = kind,
                attrs = report.attrs,
            )
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
            MDC.put(MDC_RQ_ID, current.traceId)
            MDC.put(MDC_EXT_RQ_ID, current.spanId)
        } else {
            MDC.remove(MDC_RQ_ID)
            MDC.remove(MDC_EXT_RQ_ID)
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
            message?.let { attrs[ATTR_ERROR_MESSAGE] = it }
        }
    }

    companion object {

        const val MDC_RQ_ID = "rqId"
        const val MDC_EXT_RQ_ID = "extRqId"
        const val MDC_CLIENT = "client"
        const val MDC_SERVER = "server"

        // Стандартные заголовки (Legacy)
        const val HEADER_X_SENDER = "x-sender"

        // W3C Trace Context (Стандарт OTel)
        const val HEADER_TRACEPARENT = "traceparent"
        const val HEADER_BAGGAGE = "baggage"
        const val TRACEPARENT_PREFIX = "00-"
        const val TRACEPARENT_DELIMITER = "-"

        /**
         * Количество наносекунд в одной секунде
         */
        private const val NANOS_PER_SECOND = 1_000_000_000L

        /**
         * Парсит заголовок багажа согласно W3C (key=value,key2=value2)
         */
        fun parseBaggage(header: String?): Map<String, String> {
            if (header.isNullOrBlank()) {
                return emptyMap()
            }
            return header.split(",")
                .map { it.substringBefore(";").trim() } // OTel поддерживает параметры после ; но нам пока не надо
                .filter { it.contains("=") }
                .associate {
                    val parts = it.split("=", limit = 2)
                    parts[0].trim().lowercase() to parts[1].trim()
                }
        }

        /**
         * Сериализует мапу в строку для заголовка baggage
         */
        fun formatBaggage(baggage: Map<String, String>): String? {
            if (baggage.isEmpty()) {
                return null
            }
            return baggage.entries.joinToString(",") { "${it.key}=${it.value}" }
        }
    }
}
