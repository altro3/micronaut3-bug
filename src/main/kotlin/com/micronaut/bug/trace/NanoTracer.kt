package com.micronaut.bug.trace

import com.micronaut.bug.trace.NanoTraceFilter.Companion.HEADER_API_KEY
import com.micronaut.bug.trace.TraceIdGenerator.generate
import com.micronaut.bug.trace.TraceIdGenerator.generateSpanId
import com.micronaut.bug.trace.config.TraceProperties
import com.micronaut.bug.trace.otlp.TraceExporter
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import kotlinx.coroutines.withContext
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.CONTENT_LENGTH
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpHeaders.HOST
import org.springframework.http.HttpHeaders.USER_AGENT
import org.springframework.http.HttpMethod
import reactor.util.context.ContextView
import java.time.Instant
import java.util.Deque
import java.util.concurrent.ThreadLocalRandom

class NanoTracer(
    @PublishedApi
    internal val exporter: TraceExporter,
    private val traceProps: TraceProperties,
) {
    @PublishedApi
    internal val internalStack = ThreadLocal<Deque<TraceContext>>.withInitial { ArrayDeque<TraceContext>() }

    /**
     * Статическое смещение между системным временем и монотонным счетчиком наносекунд.
     * Вычисляется один раз при создании экземпляра NanoTracer.
     * Обычная [val] работает быстрее, чем [by lazy], так как не требует синхронизации потоков.
     */
    private val clockOffsetNanos: Long

    init {
        val now = Instant.now()
        clockOffsetNanos = (now.epochSecond * NANOS_PER_SECOND + now.nano) - System.nanoTime()
    }

    @PublishedApi
    internal fun getCurrentEpochNanos(): Long = System.nanoTime() + clockOffsetNanos

    fun currentContext(): TraceContext? = internalStack.get().firstOrNull()

    fun getTraceParent(): String? {
        val current = currentContext() ?: return null
        val sb = tlStringBuilder.get().apply { setLength(0) }

        sb.append(TRACEPARENT_PREFIX)
            .append(current.traceId)
            .append('-')
            .append(current.spanId)
            .append('-')
            .append(if (current.sampled) TRACE_FLAG_SAMPLED else TRACE_FLAG_NOT_SAMPLED)

        return sb.toString()
    }

    fun startTrace(
        name: String,
        remoteTraceId: String? = null,
        remoteParentId: String? = null,
        sampled: Boolean = true,
        baggage: Map<String, String>? = null,
        propagationHeaders: Map<String, String>? = null,
        traceState: String? = null,
    ) = pushAndSync(
        TraceContext(
            traceId = remoteTraceId ?: generate(),
            spanId = generateSpanId(),
            parentId = remoteParentId, // Если он пришел, мы станем вложенным трейсом
            name = name,
            startEpochNanos = getCurrentEpochNanos(),
            sampled = sampled,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            traceState = traceState,
        )
    )

    fun startSpan(name: String, parent: TraceContext? = null): TraceContext {
        // Если родитель передан явно (из Reactor), берем его.
        // Если нет — пытаемся взять из ThreadLocal стека.
        val effectiveParent = parent ?: currentContext() ?: return startTrace(name)

        return pushAndSync(
            TraceContext(
                traceId = effectiveParent.traceId, // Наследуем traceId
                spanId = generateSpanId(),
                parentId = effectiveParent.spanId,
                name = name,
                startEpochNanos = getCurrentEpochNanos(),
                sampled = effectiveParent.sampled,
                baggage = effectiveParent.baggage,
                propagationHeaders = effectiveParent.propagationHeaders,
                traceState = effectiveParent.traceState,
            )
        )
    }

    fun stop(
        ctx: TraceContext,
        status: StatusCode = StatusCode.STATUS_CODE_OK,
        kind: SpanKind = SpanKind.SPAN_KIND_SERVER,
        attrs: Map<String, Any>? = null,
        forceExport: Boolean = false,
    ) {

        val stack = internalStack.get()
        // Удаляем именно этот контекст (даже если закрыли не в том порядке)
        if (!stack.isEmpty() && stack.firstOrNull() === ctx) {
            stack.removeFirst()
        } else {
            stack.remove(ctx)
        }
        // 1. Проверяем, является ли спан ошибочным
        val isError = ctx.error != null || status == StatusCode.STATUS_CODE_ERROR

        // Решение об экспорте:
        // 1. Либо это принудительный экспорт (ошибка, медленный запрос)
        // 2. Либо попали в рандом по sampleRate
        val shouldExport = when {
            isError || forceExport -> true // Всегда шлем ошибки и медленные
            !ctx.sampled -> false          // Если родитель (Istio) сказал "не надо" — уважаем
            else -> ThreadLocalRandom.current().nextDouble() < traceProps.sampleRate // Иначе наше сэмплирование
        }

        if (!shouldExport) {
            syncMdc()
            return
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
            userAttrs = attrs,
            baggage = ctx.baggage,
            propagationHeaders = ctx.propagationHeaders,
            error = ctx.error,
        )
        syncMdc()
    }

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

    @PublishedApi
    internal fun syncMdc() {
        val current = internalStack.get().firstOrNull()
        if (current != null) {
            val oldTrace = MDC.get(MDC_TRACE_ID)
            if (oldTrace != current.traceId) {
                MDC.put(MDC_TRACE_ID, current.traceId)
            }

            val oldSpan = MDC.get(MDC_SPAN_ID)
            if (oldSpan != current.spanId) {
                MDC.put(MDC_SPAN_ID, current.spanId)
            }

            val oldFlags = MDC.get(MDC_TRACE_FLAGS)
            val currentFlags = if (current.sampled) TRACE_FLAG_SAMPLED else TRACE_FLAG_NOT_SAMPLED
            if (oldFlags != currentFlags) {
                MDC.put(MDC_TRACE_FLAGS, currentFlags)
            }
        } else {
            MDC.remove(MDC_TRACE_ID)
            MDC.remove(MDC_SPAN_ID)
            MDC.remove(MDC_TRACE_FLAGS) // Не забываем очищать при пустом стеке
        }
    }

    class TraceReport(
        var status: StatusCode = StatusCode.STATUS_CODE_OK
    ) {
        // Инициализируем null вместо немедленного создания мапы
        var attrs: MutableMap<String, Any>? = null
            private set

        /**
         * Ленивое наполнение атрибутов
         */
        fun putAttr(key: String, value: Any) {
            if (attrs == null) {
                attrs = HashMap(4) // Создаем обычный быстрый HashMap по требованию
            }
            attrs!![key] = value
        }

        /**
         * Удобный метод для быстрой пометки спана как ошибочного.
         */
        fun error(message: String?) {
            status = StatusCode.STATUS_CODE_ERROR
            message?.let { putAttr(ATTR_ERROR_MESSAGE, it) }
        }
    }

    companion object {

        const val MDC_TRACE_ID = "traceId"
        const val MDC_TRACE_FLAGS = "traceFlags"
        const val MDC_SPAN_ID = "spanId"
        const val MDC_SOURCE = "source"
        const val MDC_TARGET = "target"
        const val MDC_SUB_TITLE = "subTitle"
        const val MDC_MAIN_STAT = "mainstat"
        const val MDC_COLOR = "color"

        // Стандартные заголовки (Legacy)
        const val HEADER_X_SENDER = "x-sender"

        // W3C Trace Context (Стандарт OTel)
        const val HEADER_TRACEPARENT = "traceparent"
        const val HEADER_BAGGAGE = "baggage"
        const val HEADER_TRACESTATE = "tracestate"
        const val TRACEPARENT_PREFIX = "00-"

        /**
         * Флаги сэмплирования согласно спецификации W3C Trace Context.
         * "01" — трейс сэмплируется (записывается).
         * "00" — трейс не сэмплируется.
         */
        const val TRACE_FLAG_SAMPLED = "01"
        const val TRACE_FLAG_NOT_SAMPLED = "00"

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

        private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(64) }

        /**
         * Стандартные ключи атрибутов OpenTelemetry
         */
        // Resource (Service)
        const val ATTR_SERVICE_NAME = "service.name"
        const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"

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

        val METHODS_WITHOUT_BODY = setOf(
            HttpMethod.GET.name(),
            HttpMethod.HEAD.name(),
            HttpMethod.OPTIONS.name(),
            HttpMethod.DELETE.name(),
            HttpMethod.TRACE.name(),
        )

        val SENSITIVE_HEADERS = setOf(
            HttpHeaders.AUTHORIZATION.lowercase(),
            HttpHeaders.COOKIE.lowercase(),
            HttpHeaders.SET_COOKIE.lowercase(),
            HEADER_API_KEY.lowercase(),
        )

        const val MASK = "***"

        /**
         * Список для маскировки чувствительных заголовков в формате OTel (string[]).
         */
        val MASKED_VALUES = listOf(MASK)

        /**
         * Парсит заголовок багажа согласно W3C (key=value,key2=value2)
         */
        fun parseBaggage(header: String?): Map<String, String>? {
            if (header.isNullOrBlank()) {
                return null
            }

            val map = HashMap<String, String>()
            var start = 0
            val len = header.length

            while (start < len) {
                var end = header.indexOf(',', start)
                if (end == -1) {
                    end = len
                }

                val equalsIdx = header.indexOf('=', start)
                if (equalsIdx != -1 && equalsIdx < end) {
                    val semicolonIdx = header.indexOf(';', start)
                    val valueEnd = if (semicolonIdx != -1 && semicolonIdx < end) semicolonIdx else end

                    val key = header.substring(start, equalsIdx).trim().lowercase()
                    val value = header.substring(equalsIdx + 1, valueEnd).trim()

                    if (key.isNotEmpty()) {
                        map[key] = value
                    }
                }
                start = end + 1
            }
            return map
        }

        /**
         * Сериализует мапу в строку для заголовка baggage
         */
        fun formatBaggage(baggage: Map<String, String>?): String? {
            if (baggage.isNullOrEmpty()) {
                return null
            }

            val sb = StringBuilder(128)
            var isFirst = true

            // Прямой цикл по мапе не создаёт итераторов в Java 21+
            baggage.forEach { (key, value) ->
                if (!isFirst) {
                    sb.append(',')
                }
                sb.append(key).append('=').append(value)
                isFirst = false
            }

            return sb.toString()
        }
    }

    /**
     * Извлекает контекст трассировки из хранилища Project Reactor.
     *
     * В реактивном окружении (Spring Cloud Gateway, WebClient) стандартные механизмы
     * на базе ThreadLocal не работают из-за постоянной смены потоков исполнения.
     * Данный метод позволяет безопасно получить [TraceContext] из [ContextView],
     * обеспечивая непрерывность цепочки спанов внутри реактивных операторов.
     *
     * @param reactorContext неизменяемое представление контекста Reactor (ContextView).
     * @return [TraceContext], если он был ранее помещен в контекст (например, фильтром Gateway),
     * или null, если информация о трассировке отсутствует.
     */
    fun currentContext(reactorContext: ContextView): TraceContext? =
        reactorContext.getOrDefault(TraceContext::class.java, null)
}
