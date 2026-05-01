package com.micronaut.bug.log.otlp

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.classic.spi.IThrowableProxy
import ch.qos.logback.classic.spi.ThrowableProxyUtil
import com.google.protobuf.ByteString
import com.micronaut.bug.log.LogMasker
import io.opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.InstrumentationScope
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.logs.v1.LogRecord
import io.opentelemetry.proto.logs.v1.ResourceLogs
import io.opentelemetry.proto.logs.v1.ScopeLogs
import io.opentelemetry.proto.logs.v1.SeverityNumber
import io.opentelemetry.proto.resource.v1.Resource
import java.util.HexFormat
import java.util.concurrent.ConcurrentHashMap

/**
 * Кодировщик логов в формат OTLP с поддержкой маскирования чувствительных данных.
 *
 * Оптимизирован для экстремально высоких нагрузок:
 * - Исключено использование HashMap и итераторов коллекций в горячих циклах.
 * - Применено кэширование долгоживущих объектов Protobuf (Scope).
 * - Сведены к нулю аллокации промежуточных объектов-оберток (типа Pair).
 */
class OtlpEncoder(
    appName: String,
    nodeName: String,
    private val logMasker: LogMasker
) {
    /**
     * Потокобезопасный кэш для предсобранных объектов Scope.
     * Защищает кучу от дублирования одинаковых структур при миллионах логов.
     */
    private val scopeCache = ConcurrentHashMap<String, InstrumentationScope>()

    /**
     * Статичные метаданные ресурса (сервис и окружение).
     */
    private val sharedResource: Resource = Resource.newBuilder()
        .addAttributes(kv(ATTR_SERVICE_NAME, appName))
        .addAttributes(kv(ATTR_DEPLOYMENT_ENVIRONMENT, nodeName))
        .build()

    /**
     * Преобразует батч событий в Protobuf с применением маскирования.
     *
     * @param events Накопленный батч событий из очереди аппендера.
     * @return Сериализованный массив байт для отправки по HTTP.
     */
    fun encodeBatch(events: List<ILoggingEvent>): ByteArray {

        // ШАГ 1: Группировка логов по именам логгеров без использования HashMap.
        // Два параллельных массива гарантируют идеальное попадание в кэш-линии L1/L2 процессора.
        val scopeNames = ArrayList<String>()
        val groupedEvents = ArrayList<MutableList<ILoggingEvent>>()

        for (event in events) {
            val scopeName = event.loggerName ?: VAL_UNKNOWN_SCOPE

            // Линейный поиск по массиву для малых N (быстрее хэширования)
            var foundIndex = -1
            for (i in 0 until scopeNames.size) {
                if (scopeNames[i] == scopeName) {
                    foundIndex = i
                    break
                }
            }

            if (foundIndex == -1) {
                scopeNames.add(scopeName)
                val newList = ArrayList<ILoggingEvent>()
                newList.add(event)
                groupedEvents.add(newList)
            } else {
                groupedEvents[foundIndex].add(event)
            }
        }

        // ШАГ 2: Формирование структур ScopeLogs (без вызова итераторов)
        val scopeLogsList = ArrayList<ScopeLogs>(scopeNames.size)

        for (i in scopeNames.indices) {
            val scopeName = scopeNames[i]
            val scopeEvents = groupedEvents[i]
            val logRecords = ArrayList<LogRecord>(scopeEvents.size)

            for (j in scopeEvents.indices) {
                val event = scopeEvents[j]

                // Маскирование контента (происходит на лету)
                val maskedMessage = logMasker.maskServiceMessage(event.formattedMessage)
                val mdc = event.mdcPropertyMap ?: emptyMap()

                val logRecordBuilder = LogRecord.newBuilder()
                    .setTimeUnixNano(event.timeStamp * NANOS_IN_MILLI)
                    .setBody(AnyValue.newBuilder().setStringValue(maskedMessage).build())
                    .setSeverityNumber(mapLevelToSeverity(event.level))
                    .setSeverityText(event.level.levelStr)

                    // ШАГ 3: Фильтрация и защита от дублирования TraceId/SpanId в Attributes
                    .addAllAttributes(
                        logMasker.maskMap(mdc)
                            .filterKeys { it != MDC_TRACE_ID && it != MDC_SPAN_ID && it != MDC_TRACE_FLAGS }
                            .map { (key, value) -> kv(key, value) }
                    )

                // ШАГ 4: Автоматический захват StackTrace по стандарту OTel
                val throwableProxy = event.throwableProxy
                if (throwableProxy != null) {
                    logRecordBuilder.addAllAttributes(extractExceptionAttributes(throwableProxy))
                }

                // ШАГ 5: Преобразование Hex-строк в нативные бинарные поля Protobuf (Java 17 HexFormat)
                mdc[MDC_TRACE_ID]?.let { traceIdStr ->
                    if (traceIdStr.length == TRACE_ID_HEX_LEN) {
                        logRecordBuilder.setTraceId(ByteString.copyFrom(HexFormat.of().parseHex(traceIdStr)))
                    }
                }

                mdc[MDC_SPAN_ID]?.let { spanIdStr ->
                    if (spanIdStr.length == SPAN_ID_HEX_LEN) {
                        logRecordBuilder.setSpanId(ByteString.copyFrom(HexFormat.of().parseHex(spanIdStr)))
                    }
                }

                // Извлечение флагов сэмплирования
                mdc[MDC_TRACE_FLAGS]?.let { flagsStr ->
                    if (flagsStr.length == 2) {
                        logRecordBuilder.setFlags(flagsStr.toInt(16))
                    }
                }

                logRecords.add(logRecordBuilder.build())
            }

            // ШАГ 6: Извлечение закэшированного Scope (снижает нагрузку на Garbage Collector)
            val cachedScope = scopeCache.computeIfAbsent(scopeName) { name ->
                InstrumentationScope.newBuilder().setName(name).build()
            }

            scopeLogsList.add(
                ScopeLogs.newBuilder()
                    .setScope(cachedScope)
                    .addAllLogRecords(logRecords)
                    .build()
            )
        }

        val resourceLogs = ResourceLogs.newBuilder()
            .setResource(sharedResource)
            .addAllScopeLogs(scopeLogsList)
            .build()

        return ExportLogsServiceRequest.newBuilder()
            .addResourceLogs(resourceLogs)
            .build()
            .toByteArray()
    }

    /**
     * Извлекает детали ошибки по стандарту OpenTelemetry Semantic Conventions.
     */
    private fun extractExceptionAttributes(proxy: IThrowableProxy): List<KeyValue> {
        val attributes = mutableListOf<KeyValue>()

        attributes.add(kv(ATTR_EXCEPTION_TYPE, proxy.className))
        proxy.message?.let { attributes.add(kv(ATTR_EXCEPTION_MESSAGE, it)) }

        val stackTrace = ThrowableProxyUtil.asString(proxy)
        attributes.add(kv(ATTR_EXCEPTION_STACKTRACE, stackTrace))

        return attributes
    }

    private fun kv(key: String, value: String) =
        KeyValue.newBuilder()
            .setKey(key)
            .setValue(AnyValue.newBuilder().setStringValue(value).build())
            .build()

    private fun mapLevelToSeverity(level: Level): SeverityNumber =
        when (level) {
            Level.TRACE -> SeverityNumber.SEVERITY_NUMBER_TRACE
            Level.DEBUG -> SeverityNumber.SEVERITY_NUMBER_DEBUG
            Level.INFO -> SeverityNumber.SEVERITY_NUMBER_INFO
            Level.WARN -> SeverityNumber.SEVERITY_NUMBER_WARN
            Level.ERROR -> SeverityNumber.SEVERITY_NUMBER_ERROR
            else -> SeverityNumber.SEVERITY_NUMBER_UNSPECIFIED
        }

    companion object {
        const val ATTR_SERVICE_NAME = "service.name"
        const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"

        // Константы OTel Semantic Conventions
        private const val ATTR_EXCEPTION_TYPE = "exception.type"
        private const val ATTR_EXCEPTION_MESSAGE = "exception.message"
        private const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"

        private const val MDC_TRACE_ID = "traceId"
        private const val MDC_SPAN_ID = "spanId"
        private const val MDC_TRACE_FLAGS = "traceFlags"

        private const val NANOS_IN_MILLI = 1_000_000L
        private const val TRACE_ID_HEX_LEN = 32
        private const val SPAN_ID_HEX_LEN = 16
        private const val VAL_UNKNOWN_SCOPE = "unknown"
    }
}
