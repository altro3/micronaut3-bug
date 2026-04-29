package com.micronaut.bug.trace

import com.micronaut.bug.trace.TraceIdGenerator.generate
import com.micronaut.bug.trace.TraceIdGenerator.generateSpanId
import com.micronaut.bug.trace.config.TraceProperties
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import kotlinx.coroutines.withContext
import org.slf4j.MDC
import org.springframework.http.HttpHeaders.CONTENT_LENGTH
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpHeaders.HOST
import org.springframework.http.HttpHeaders.USER_AGENT
import java.time.Instant
import java.util.Deque
import java.util.concurrent.ThreadLocalRandom

class NanoTracer(
    @PublishedApi
    internal val exporter: TempoExporter,
    private val traceProps: TraceProperties,
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
     * @param traceState Сырая строка из заголовка "tracestate" для обеспечения совместимости.
     * @return Созданный и помещенный в стек [TraceContext].
     */
    fun startTrace(
        name: String,
        remoteTraceId: String? = null,
        remoteParentId: String? = null,
        baggage: Map<String, String> = emptyMap(),
        propagationHeaders: Map<String, String> = emptyMap(),
        traceState: String? = null,
    ) = pushAndSync(
        TraceContext(
            traceId = remoteTraceId ?: generate(),
            spanId = generateSpanId(),
            parentId = remoteParentId, // Если он пришел, мы станем вложенным трейсом
            name = name,
            startEpochNanos = getCurrentEpochNanos(),
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            traceState = traceState,
        )
    )

    /**
     * Создает и активирует дочерний спан, наследуя контекст от текущего активного спана.
     * Если активный контекст отсутствует, метод автоматически создает корневой трейс.
     *
     * Наследует [TraceContext.baggage], [TraceContext.propagationHeaders] и
     * [TraceContext.traceState] от родителя. Если родитель отсутствует в текущем
     * потоке, инициализирует новый трейс.
     *
     * @param name Имя вложенной операции (например, "SERVICE: processOrder" или "CLIENT: callPaymentApi").
     * @return Новый [TraceContext], ставший вершиной стека в текущем потоке.
     */
    fun startSpan(name: String): TraceContext {
        val parent = currentContext()
        if (parent == null) {
            return startTrace(name)
        }

        return pushAndSync(
            TraceContext(
                traceId = parent.traceId, // Наследуем traceId
                spanId = generateSpanId(),
                parentId = parent.spanId,
                name = name,
                startEpochNanos = getCurrentEpochNanos(),
                baggage = parent.baggage,
                propagationHeaders = parent.propagationHeaders,
                traceState = parent.traceState,
            )
        )
    }

    /**
     * Завершает операцию и отправляет данные в экспортер.
     *
     * @param ctx контекст завершаемого спана.
     * @param status статус завершения (OK/ERROR).
     * @param attrs дополнительные метаданные.
     * @param forceExport если true — игнорирует [sampleRate] и всегда отправляет спан (для ошибок и тормозов).
     */
    fun stop(
        ctx: TraceContext,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        kind: SpanKind = SpanKind.SPAN_KIND_SERVER,
        attrs: Map<String, Any> = emptyMap(),
        forceExport: Boolean = false,
    ) {

        val stack = internalStack.get()
        // Удаляем именно этот контекст (даже если закрыли не в том порядке)
        stack.remove(ctx)
        // 1. Проверяем, является ли спан ошибочным
        val isError = ctx.error != null || status == StatusCode.STATUS_CODE_ERROR

        // Решение об экспорте:
        // 1. Либо это принудительный экспорт (ошибка, медленный запрос)
        // 2. Либо попали в рандом по sampleRate
        if (!isError && !forceExport && ThreadLocalRandom.current().nextDouble() >= traceProps.sampleRate) {
            syncMdc()
            return
        }

        // Объединяем пользовательские атрибуты с метаданными контекста
        val finalAttrs = attrs.toMutableMap().apply {

            // Если в контексте зафиксирована ошибка, вытаскиваем её данные
            ctx.error?.let {
                put(ATTR_EXCEPTION_TYPE, it.javaClass.name)
                put(ATTR_EXCEPTION_MESSAGE, it.message ?: it.javaClass.simpleName)
                put(ATTR_EXCEPTION_STACKTRACE, it.stackTraceToString())
            }

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
            endEpochNanos = getCurrentEpochNanos(),
            status = status,
            kind = kind,
            attrs = finalAttrs,
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
            return withContext(TraceElement(internalStack.get(), this)) {
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

    /**
     * Создает элемент контекста корутины для проброса текущего состояния трейсинга.
     * Используется при ручном запуске корутин (launch, async).
     */
    fun dispatcher(): TraceElement {
        // Делаем копию текущего стека из ThreadLocal
        val snapshot = internalStack.get()?.let { ArrayDeque(it) } ?: ArrayDeque()
        return TraceElement(snapshot, this)
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
        const val HEADER_TRACESTATE = "tracestate"
        const val TRACEPARENT_PREFIX = "00-"
        const val TRACEPARENT_DELIMITER = "-"

        val OTEL_MAPPED_HEADERS = setOf(
            CONTENT_LENGTH.lowercase(),
            CONTENT_TYPE.lowercase(),
            USER_AGENT.lowercase(),
            // Сюда можно добавить host, так как он уходит в server.address
            HOST.lowercase(),
            // Добавляем сюда заголовки трейсинга, чтобы они не попадали в атрибуты
            HEADER_TRACEPARENT.lowercase(),
            HEADER_TRACESTATE.lowercase(),
            HEADER_BAGGAGE.lowercase(), // Багаж мы и так пишем отдельно с префиксом "baggage."
        )

        /**
         * Стандартные ключи атрибутов OpenTelemetry
         */
        // Resource (Service)
        const val ATTR_SERVICE_NAME = "service.name"

        const val ATTR_SERVER_ADDRESS = "server.address"
        const val ATTR_SERVER_PORT = "server.port"
        const val ATTR_CLIENT_ADDRESS = "client.address"

        // HTTP Request
        const val ATTR_HTTP_REQUEST_METHOD = "http.request.method"
        const val ATTR_HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
        const val PREFIX_HTTP_REQUEST_HEADER = "http.request.header."

        const val ATTR_URL_FULL = "url.full"
        const val ATTR_URL_SCHEME = "url.scheme"
        const val ATTR_URL_PATH = "url.path"
        const val ATTR_URL_QUERY = "url.query"
        const val ATTR_USER_AGENT_ORIGINAL = "user_agent.original"

        // HTTP Response
        const val ATTR_HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
        const val ATTR_HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"
        const val PREFIX_HTTP_RESPONSE_HEADER = "http.response.header."

        // Exceptions
        const val ATTR_EXCEPTION_TYPE = "exception.type"
        const val ATTR_EXCEPTION_MESSAGE = "exception.message"
        const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"

        const val ATTR_CLIENT = "client"
        const val ATTR_SERVER = "server"
        const val ATTR_PEER_SERVICE = "peer.service"

        /**
         * Кастомные ключи для логгинг-фильтров
         */
        const val ATTR_ERROR_MESSAGE = "error.message"
        const val ATTR_ERROR_TYPE = "error.type"
        const val PREFIX_BAGGAGE = "baggage."
        const val PREFIX_PROPAGATION = "prop."

        /**
         * Флаг аномально медленного запроса.
         * Позволяет быстро отфильтровать трейсы с задержкой выше установленного порога.
         */
        const val ATTR_HTTP_SLOW_REQUEST = "http.slow_request"

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
