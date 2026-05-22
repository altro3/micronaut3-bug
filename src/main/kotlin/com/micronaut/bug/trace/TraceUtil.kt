package com.micronaut.bug.trace

import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod

object TraceUtil {

    // Системные константы для интеграции с MDC (SLF4J)
    const val MDC_TRACE_ID = "traceId"
    const val MDC_SPAN_ID = "spanId"
    const val MDC_TRACE_FLAGS = "traceFlags"
    const val MDC_SOURCE = "source"
    const val MDC_TARGET = "target"

    // Флаги сэмплирования по спецификации W3C Trace Context
    const val TRACE_FLAG_SAMPLED = "01"
    const val TRACE_FLAG_NOT_SAMPLED = "00"

    // Префиксы и заголовки W3C / OpenTelemetry
    const val TRACEPARENT_PREFIX = "00-"
    const val HEADER_TRACEPARENT = "traceparent"
    const val HEADER_TRACESTATE = "tracestate"
    const val HEADER_BAGGAGE = "baggage"
    const val HEADER_X_SENDER = "x-sender"

    // Константы времени
    const val NANOS_PER_SECOND = 1_000_000_000L

    // Атрибуты OpenTelemetry (Семантические конвенции v1.20+)
    const val ATTR_SERVICE_NAME = "service.name"
    const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"
    const val ATTR_CLIENT = "client"
    const val ATTR_SERVER = "server"

    const val ATTR_URL_FULL = "url.full"
    const val ATTR_URL_SCHEME = "url.scheme"
    const val ATTR_URL_PATH = "url.path"
    const val ATTR_URL_QUERY = "url.query"

    const val ATTR_HTTP_REQUEST_METHOD = "http.request.method"
    const val ATTR_HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
    const val ATTR_HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"
    const val ATTR_HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
    const val ATTR_HTTP_SLOW_REQUEST = "http.slow_request"
    const val ATTR_USER_AGENT_ORIGINAL = "user_agent.original"
    const val ATTR_SERVER_ADDRESS = "server.address"
    const val ATTR_SERVER_PORT = "server.port"
    const val ATTR_CLIENT_ADDRESS = "client.address"

    // Константы для обработки ошибок и исключений
    const val ATTR_EXCEPTION_TYPE = "exception.type"
    const val ATTR_EXCEPTION_MESSAGE = "exception.message"
    const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"
    const val ATTR_ERROR_MESSAGE = "error.message"

    // Префиксы для кастомных заголовков HTTP
    const val PREFIX_HTTP_REQUEST_HEADER = "http.request.header."
    const val PREFIX_HTTP_RESPONSE_HEADER = "http.response.header."
    const val PREFIX_BAGGAGE = "baggage."
    const val PREFIX_PROPAGATION = "propagation."
    const val MASK = "***"

    // Маскирование и фильтрация приватных данных
    @JvmField
    val MASKED_VALUES = listOf(MASK)

    @JvmField
    val OTEL_MAPPED_HEADERS = setOf(
        HttpHeaders.CONTENT_LENGTH.lowercase(),
        HttpHeaders.CONTENT_TYPE.lowercase(),
        HttpHeaders.USER_AGENT.lowercase(),
        HttpHeaders.HOST.lowercase(),
        HEADER_TRACEPARENT.lowercase(),
        HEADER_TRACESTATE.lowercase(),
        HEADER_BAGGAGE.lowercase(),
    )

    @JvmField
    val METHODS_WITHOUT_BODY = setOf(
        HttpMethod.GET.name(),
        HttpMethod.HEAD.name(),
        HttpMethod.OPTIONS.name(),
        HttpMethod.DELETE.name(),
        HttpMethod.TRACE.name(),
    )

    @JvmField
    val SENSITIVE_HEADERS = setOf(
        HttpHeaders.AUTHORIZATION.lowercase(),
        HttpHeaders.COOKIE.lowercase(),
        HttpHeaders.SET_COOKIE.lowercase(),
        NanoTraceFilter.HEADER_API_KEY,
    )

    /**
     * Высокопроизводительный разбор заголовка Baggage (W3C).
     * Разбирает строку формата `key1=value1,key2=value2;metadata` за один проход.
     * Игнорирует опциональные свойства (properties/metadata) после точки с запятой.
     */
    fun parseBaggage(header: String): Map<String, String>? {
        if (header.isBlank()) return null

        val result = HashMap<String, String>(4)
        val len = header.length
        var i = 0

        while (i < len) {
            // Пропускаем пробелы перед ключом
            while (i < len && header[i] <= ' ') i++
            if (i >= len) break

            // Ищем конец ключа (знак '=' или конец строки)
            val keyStart = i
            while (i < len && header[i] != '=' && header[i] != ',' && header[i] != ';') i++
            val keyEnd = i

            if (i < len && header[i] == '=') {
                i++ // Пропускаем '='

                // Пропускаем пробелы перед значением
                while (i < len && header[i] <= ' ') i++
                val valStart = i

                // Ищем конец значения (запятая или точка с запятой)
                var inMetadata = false
                while (i < len) {
                    val ch = header[i]
                    if (ch == ',') {
                        break
                    } else if (ch == ';') {
                        inMetadata = true
                    }
                    i++
                }

                var valEnd = i
                if (inMetadata) {
                    // Если была точка с запятой, отматываем назад до её появления, игнорируя метаданные
                    valEnd = header.indexOf(';', valStart)
                    if (valEnd == -1) valEnd = i
                }

                // Убираем возможные шлейфовые пробелы у значения
                var actualValEnd = valEnd
                while (actualValEnd > valStart && header[actualValEnd - 1] <= ' ') actualValEnd--

                val key = header.substring(keyStart, keyEnd).trim()
                if (key.isNotEmpty() && actualValEnd > valStart) {
                    result[key] = header.substring(valStart, actualValEnd)
                }
            } else {
                // Ключ без значения или некорректная структура, пропускаем до следующей пары
                while (i < len && header[i] != ',') i++
            }
            i++ // Пропускаем запятую
        }

        return if (result.isEmpty()) null else result
    }

    fun Collection<String>.containsIgnoreCase(charSequence: String): Boolean {
        for (element in this) {
            if (element.equals(charSequence, ignoreCase = true)) return true
        }
        return false
    }
}
