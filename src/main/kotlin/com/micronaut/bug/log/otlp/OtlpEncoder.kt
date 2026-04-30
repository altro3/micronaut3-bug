package com.micronaut.bug.log.otlp

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.spi.ILoggingEvent
import com.micronaut.bug.log.LogMasker
import io.opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.logs.v1.LogRecord
import io.opentelemetry.proto.logs.v1.ResourceLogs
import io.opentelemetry.proto.logs.v1.ScopeLogs
import io.opentelemetry.proto.logs.v1.SeverityNumber
import io.opentelemetry.proto.resource.v1.Resource

/**
 * Кодировщик логов в формат OTLP с поддержкой маскирования чувствительных данных.
 *
 * Маскирование применяется только здесь (перед отправкой в сеть),
 * оставляя логи в консоли нетронутыми для удобства локальной отладки.
 */
class OtlpEncoder(
    appName: String,
    nodeName: String,
    private val logMasker: LogMasker
) {
    /**
     * Статичные метаданные ресурса.
     */
    private val sharedResource: Resource = Resource.newBuilder()
        .addAttributes(createKeyValue(ATTR_SERVICE_NAME, appName))
        .addAttributes(createKeyValue(ATTR_NODE_NAME, nodeName))
        .build()

    /**
     * Преобразует батч событий в Protobuf с применением маскирования.
     */
    fun encodeBatch(events: List<ILoggingEvent>): ByteArray {
        val logRecords = events.map { event ->
            // Маскируем текстовое сообщение (включая отчеты нашего фильтра)
            val maskedMessage = logMasker.maskServiceMessage(event.formattedMessage)

            LogRecord.newBuilder()
                .setTimeUnixNano(event.timeStamp * NANOS_IN_MILLI)
                .setBody(AnyValue.newBuilder().setStringValue(maskedMessage).build())
                .setSeverityNumber(mapLevelToSeverity(event.level))
                .setSeverityText(event.level.levelStr)
                // Используем маскирование карты для MDC (Mapped Diagnostic Context)
                .addAllAttributes(logMasker.maskMap(event.mdcPropertyMap).map { (key, value) ->
                    createKeyValue(key, value)
                })
                .build()
        }

        val resourceLogs = ResourceLogs.newBuilder()
            .setResource(sharedResource)
            .addScopeLogs(ScopeLogs.newBuilder().addAllLogRecords(logRecords).build())
            .build()

        return ExportLogsServiceRequest.newBuilder()
            .addResourceLogs(resourceLogs)
            .build()
            .toByteArray()
    }

    private fun createKeyValue(key: String, value: String) =
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
        private const val ATTR_SERVICE_NAME = "service.name"
        private const val ATTR_NODE_NAME = "node.name"
        private const val NANOS_IN_MILLI = 1_000_000L
    }
}
