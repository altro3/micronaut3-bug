package com.altro.common.log.otlp

import com.google.protobuf.ByteString
import com.altro.common.log.LogMasker
import com.altro.common.log.LogUtil.formatStackTrace
import com.altro.common.log.config.LogProperties
import com.altro.common.trace.TraceUtil.MDC_COLOR
import com.altro.common.trace.http.NanoTraceFilter.Companion.MDC_USER_ID
import io.opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.InstrumentationScope
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.logs.v1.LogRecord
import io.opentelemetry.proto.logs.v1.ResourceLogs
import io.opentelemetry.proto.logs.v1.ScopeLogs
import io.opentelemetry.proto.logs.v1.SeverityNumber
import io.opentelemetry.proto.resource.v1.Resource
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.LogEvent
import org.apache.logging.log4j.core.impl.ThrowableProxy

class OtlpLogEncoder(
    appName: String,
    nodeName: String,
    private val props: LogProperties,
    private val logMasker: LogMasker
) {

    // Массив для быстрого доступа по индексу (циклический)
    private val scopeCache = HashMap<String, InstrumentationScope>(CACHE_SIZE, 1F)
    private val scopeRingBuffer = arrayOfNulls<String>(CACHE_SIZE)

    // Указатель для ротации (заменяем самый старый элемент)
    private var ringPointer = 0

    private val sharedResource: Resource = Resource.newBuilder()
        .addAttributes(KeyValue.newBuilder().setKey(ATTR_SERVICE_NAME).setValue(AnyValue.newBuilder().setStringValue(appName).build()).build())
        .addAttributes(KeyValue.newBuilder().setKey(ATTR_DEPLOYMENT_ENVIRONMENT).setValue(AnyValue.newBuilder().setStringValue(nodeName).build()).build())
        .build()

    fun encodeBatch(events: List<LogEvent>): ByteArray {
        val size = events.size
        if (size == 0) return EMPTY_BYTE_ARRAY

        // Извлекаем переиспользуемые билдеры из ThreadLocal пула
        val requestBuilder = tlRequestBuilder.get().clear()
        val resLogsBuilder = tlResourceLogsBuilder.get().clear().setResource(sharedResource)
        val logRecordBuilder = tlLogRecordBuilder.get()

        val scopeNames = tlScopeNames.get().apply { clear() }
        val indexMap = tlIndexMap.get().apply { clear() }
        val groupedEvents = tlGroupedEvents.get().apply { clear() }

        var lastFoundIndex = -1
        var lastScopeName: String? = null

        for (i in 0 until size) {
            val event = events[i]
            val scopeName = event.loggerName ?: VAL_UNKNOWN_SCOPE

            @Suppress("StringReferentialEquality")
            var foundIndex = if (scopeName === lastScopeName) {
                lastFoundIndex
            } else {
                indexMap.getOrDefault(scopeName, -1)
            }

            if (foundIndex == -1) {
                foundIndex = scopeNames.size
                scopeNames.add(scopeName)
                indexMap[scopeName] = foundIndex

                // Проверяем, есть ли в пуле готовый список для этой позиции
                if (foundIndex >= groupedEvents.size) {
                    groupedEvents.add(ArrayList(size)) // Новая аллокация только при первом росте
                } else {
                    groupedEvents[foundIndex].clear() // Переиспользуем существующий
                }
            }

            groupedEvents[foundIndex].add(event)

            lastFoundIndex = foundIndex
            lastScopeName = scopeName
        }

        // ШАГ 2: Формирование структур ScopeLogs
        val scopeNamesSize = scopeNames.size
        for (i in 0 until scopeNamesSize) {
            val scopeName = scopeNames[i]
            val scopeEvents = groupedEvents[i]

            val cachedScope = getOrCreateScope(scopeName)
            val scopeLogsBuilder = tlScopeLogsBuilder.get().clear()
            scopeLogsBuilder.scope = cachedScope

            val scopeEventsSize = scopeEvents.size

            // ШАГ 3: Заполнение LogRecord'ов
            for (j in 0 until scopeEventsSize) {
                val event = scopeEvents[j]

                // Обязательный сброс состояния предыдущего лога в переиспользуемом билдере
                logRecordBuilder.clear()

                // Маскирование тела лога перед упаковкой
                val maskedMessage = logMasker.maskServiceMessage(event.message.formattedMessage)
                logRecordBuilder.bodyBuilder.stringValue = maskedMessage
                val instant = event.instant
                val nanoTs = instant.epochSecond * NANOS_IN_SEC + instant.nanoOfSecond

                logRecordBuilder
                    .setTimeUnixNano(nanoTs)
                    .setSeverityNumber(mapLevelToSeverity(event.level))
                    .setSeverityText(event.level.name())

                // ШАГ 4: Перенос MDC через ReadOnlyStringMap (Garbage-Free обход) и связывание с трассировкой
                var colorFound = false
                val mdc = event.contextData
                if (!mdc.isEmpty) {
                    mdc.forEach { key: String, value: Any? ->

                        val stringValue = value as? String ?: return@forEach
                        when (key) {
                            MDC_TRACE_ID -> {
                                if (stringValue.length == TRACE_ID_HEX_LEN) {
                                    val bytes = parseHexToBytes(stringValue, TRACE_ID_BYTES_LEN)
                                    logRecordBuilder.traceId = ByteString.copyFrom(bytes, 0, TRACE_ID_BYTES_LEN)
                                }
                            }

                            MDC_SPAN_ID -> {
                                if (stringValue.length == SPAN_ID_HEX_LEN) {
                                    val bytes = parseHexToBytes(stringValue, SPAN_ID_BYTES_LEN)
                                    logRecordBuilder.spanId = ByteString.copyFrom(bytes, 0, SPAN_ID_BYTES_LEN)
                                }
                            }

                            MDC_TRACE_FLAGS -> {
                                if (stringValue.length == 2) {
                                    logRecordBuilder.flags = stringValue.toInt(16)
                                }
                            }

                            MDC_COLOR -> {
                                colorFound = true
                                logRecordBuilder.addAttributes(keyValue(ATTR_COLOR, stringValue))
                            }

                            MDC_USER_ID -> {
                                logRecordBuilder.addAttributes(keyValue(ATTR_USER_ID, stringValue))
                            }

                            else -> {
                                logRecordBuilder.addAttributes(keyValue(key, stringValue))
                            }
                        }
                    }
                    if (!colorFound) {
                        logRecordBuilder.addAttributes(keyValue(ATTR_COLOR, "green"))
                    }
                    logRecordBuilder.addAttributes(keyValue(ATTR_ID, nanoTs))
                }

                val throwableProxy = event.thrownProxy
                if (throwableProxy != null) {
                    fillExceptionAttributes(logRecordBuilder, throwableProxy)
                }

                scopeLogsBuilder.addLogRecords(logRecordBuilder.build())
            }

            resLogsBuilder.addScopeLogs(scopeLogsBuilder.build())
        }

        // ВАЖНО: В самом конце метода, перед return, очищаем вложенные списки
        for (i in scopeNames.indices) {
            groupedEvents[i].clear()
        }

        // ШАГ 6: Финальная сериализация в байты
        return requestBuilder
            .addResourceLogs(resLogsBuilder.build())
            .build()
            .toByteArray()
    }

    private fun getOrCreateScope(name: String): InstrumentationScope {
        // 1. Пытаемся найти в кэше
        val existing = scopeCache[name]
        if (existing != null) return existing

        // 2. Если кэш полон, выселяем самого старого по указателю ringPointer
        if (scopeCache.size >= CACHE_SIZE) {
            val oldName = scopeRingBuffer[ringPointer]
            if (oldName != null) {
                scopeCache.remove(oldName)
            }
        }

        // 3. Создаем новый Scope
        val newScope = InstrumentationScope.newBuilder().setName(name).build()

        // 4. Записываем в кэш и в кольцевой буфер
        scopeCache[name] = newScope
        scopeRingBuffer[ringPointer] = name

        // 5. Двигаем указатель по кругу
        ringPointer = (ringPointer + 1) and CACHE_MASK

        return newScope
    }

    private fun fillExceptionAttributes(logRecordBuilder: LogRecord.Builder, proxy: ThrowableProxy) {
        logRecordBuilder.addAttributes(keyValue(ATTR_EXCEPTION_TYPE, proxy.name))
        proxy.message?.let {
            logRecordBuilder.addAttributes(keyValue(ATTR_EXCEPTION_MESSAGE, it))
        }

        val sb = tlStringBuilder.get()
        sb.setLength(0)

        formatStackTrace(
            proxy = proxy,
            sb = sb,
            maxLines = props.stackTraceMaxLines,
            rootCauseFull = props.stackTraceRootCauseFull,
        )

        logRecordBuilder.addAttributes(keyValue(ATTR_EXCEPTION_STACKTRACE, sb.toString()))
    }

    private fun keyValue(k: String, v: Any): KeyValue {
        val vBuilder = tlValueBuilder.get().clear()

        when (v) {
            is String -> vBuilder.setStringValue(v)
            is Long -> vBuilder.setIntValue(v)
            is Int -> vBuilder.setIntValue(v.toLong())
            is Boolean -> vBuilder.setBoolValue(v)
            is Double -> vBuilder.setDoubleValue(v)
            is Float -> vBuilder.setDoubleValue(v.toDouble())
            is Iterable<*> -> {
                val arrayBuilder = vBuilder.arrayValueBuilder
                val itemValBuilder = tlItemValueBuilder.get()
                for (item in v) {
                    if (item != null) {
                        arrayBuilder.addValues(
                            itemValBuilder.clear()
                                .setStringValue(item.toString())
                                .build()
                        )
                    }
                }
            }

            else -> vBuilder.setStringValue(v.toString())
        }
        return tlKeyValueBuilder.get().clear()
            .setKey(k)
            .setValue(vBuilder)
            .build()
    }

    private fun mapLevelToSeverity(level: Level): SeverityNumber =
        when (level) {
            Level.TRACE -> SeverityNumber.SEVERITY_NUMBER_TRACE
            Level.DEBUG -> SeverityNumber.SEVERITY_NUMBER_DEBUG
            Level.INFO -> SeverityNumber.SEVERITY_NUMBER_INFO
            Level.WARN -> SeverityNumber.SEVERITY_NUMBER_WARN
            Level.ERROR -> SeverityNumber.SEVERITY_NUMBER_ERROR
            Level.FATAL -> SeverityNumber.SEVERITY_NUMBER_FATAL
            else -> SeverityNumber.SEVERITY_NUMBER_UNSPECIFIED
        }

    private fun parseHexToBytes(hex: String, bytesLen: Int): ByteArray {
        val target = tlByteArrayBuffer.get() // Забираем буфер прямо здесь
        for (i in 0 until bytesLen) {
            val h = Character.digit(hex[i * 2], 16)
            val l = Character.digit(hex[i * 2 + 1], 16)

            if (h == -1 || l == -1) {
                target.fill(0, 0, bytesLen)
                return target
            }
            target[i] = ((h shl 4) or l).toByte()
        }
        return target
    }

    companion object {

        const val ATTR_SERVICE_NAME = "service.name"
        const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"
        private const val ATTR_EXCEPTION_TYPE = "exception.type"
        private const val ATTR_EXCEPTION_MESSAGE = "exception.message"
        private const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"
        private const val ATTR_ID = "id"
        private const val ATTR_USER_ID = "user.id"
        private const val ATTR_COLOR = "color"

        // Ключи MDC, генерируемые механизмами трассировки
        private const val MDC_TRACE_ID = "traceId"
        private const val MDC_SPAN_ID = "spanId"
        private const val MDC_TRACE_FLAGS = "traceFlags"

        private const val NANOS_IN_SEC = 1_000_000_000L

        private const val TRACE_ID_HEX_LEN = 32
        private const val TRACE_ID_BYTES_LEN = 16

        private const val SPAN_ID_HEX_LEN = 16
        private const val SPAN_ID_BYTES_LEN = 8
        private const val VAL_UNKNOWN_SCOPE = "unknown"

        private val EMPTY_BYTE_ARRAY = ByteArray(0)

        // Обязательно степень двойки
        private const val CACHE_SIZE = 2048
        private const val CACHE_MASK = CACHE_SIZE - 1

        // THREAD_LOCAL POOLING: Исключаем аллокации билдеров Protobuf
        private val tlRequestBuilder = ThreadLocal.withInitial { ExportLogsServiceRequest.newBuilder() }
        private val tlResourceLogsBuilder = ThreadLocal.withInitial { ResourceLogs.newBuilder() }
        private val tlLogRecordBuilder = ThreadLocal.withInitial { LogRecord.newBuilder() }
        private val tlKeyValueBuilder = ThreadLocal.withInitial { KeyValue.newBuilder() }
        private val tlScopeLogsBuilder = ThreadLocal.withInitial { ScopeLogs.newBuilder() }
        private val tlScopeNames = ThreadLocal.withInitial { ArrayList<String>(16) }
        private val tlIndexMap = ThreadLocal.withInitial { HashMap<String, Int>(16) }
        private val tlGroupedEvents = ThreadLocal.withInitial { ArrayList<MutableList<LogEvent>>(16) }
        private val tlValueBuilder = ThreadLocal.withInitial { AnyValue.newBuilder() }
        private val tlItemValueBuilder = ThreadLocal.withInitial { AnyValue.newBuilder() }

        // THREAD_LOCAL POOLING: Исключаем аллокации массивов байт
        // Берем максимальный размер (16 байт для 32-символьного traceId)
        private val tlByteArrayBuffer = ThreadLocal.withInitial { ByteArray(TRACE_ID_BYTES_LEN) }
        private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(2048) }
    }
}
