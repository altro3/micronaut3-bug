package com.micronaut.bug.trace

import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod

object TraceUtil {
    const val MDC_TRACE_ID = "traceId"
    const val MDC_TRACE_FLAGS = "traceFlags"
    const val MDC_SPAN_ID = "spanId"
    const val MDC_SOURCE = "source"
    const val MDC_TARGET = "target"
    const val MDC_SUB_TITLE = "subTitle"
    const val MDC_MAIN_STAT = "mainstat"
    const val MDC_COLOR = "color"

    const val HEADER_X_SENDER = "x-sender"
    const val HEADER_TRACEPARENT = "traceparent"
    const val HEADER_BAGGAGE = "baggage"
    const val HEADER_TRACESTATE = "tracestate"
    const val TRACEPARENT_PREFIX = "00-"

    const val TRACE_FLAG_SAMPLED = "01"
    const val TRACE_FLAG_NOT_SAMPLED = "00"

    const val ATTR_ERROR_MESSAGE = "error.message"
    const val ATTR_SERVICE_NAME = "service.name"
    const val ATTR_DEPLOYMENT_ENVIRONMENT = "deployment.environment"
    const val ATTR_SERVER_ADDRESS = "server.address"
    const val ATTR_SERVER_PORT = "server.port"
    const val ATTR_CLIENT_ADDRESS = "client.address"
    const val ATTR_HTTP_REQUEST_METHOD = "http.request.method"
    const val ATTR_HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
    const val PREFIX_HTTP_REQUEST_HEADER = "http.request.header."
    const val ATTR_URL_FULL = "url.full"
    const val ATTR_URL_SCHEME = "url.scheme"
    const val ATTR_URL_PATH = "url.path"
    const val ATTR_URL_QUERY = "url.query"
    const val ATTR_USER_AGENT_ORIGINAL = "user_agent.original"
    const val ATTR_HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
    const val ATTR_HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"
    const val PREFIX_HTTP_RESPONSE_HEADER = "http.response.header."
    const val ATTR_EXCEPTION_TYPE = "exception.type"
    const val ATTR_EXCEPTION_MESSAGE = "exception.message"
    const val ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"
    const val ATTR_CLIENT = "client"
    const val ATTR_SERVER = "server"
    const val ATTR_PEER_SERVICE = "peer.service"
    const val ATTR_ERROR_TYPE = "error.type"
    const val PREFIX_BAGGAGE = "baggage."
    const val PREFIX_PROPAGATION = "prop."
    const val ATTR_HTTP_SLOW_REQUEST = "http.slow_request"

    const val NANOS_PER_SECOND = 1_000_000_000L
    const val MASK = "***"

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
        NanoTraceFilter.HEADER_API_KEY, // Ссылка на константу из фильтра
    )

    @JvmStatic
    fun parseBaggage(header: String?): Map<String, String>? {
        if (header.isNullOrBlank()) return null
        val map = HashMap<String, String>()
        var start = 0
        val len = header.length
        while (start < len) {
            var end = header.indexOf(',', start)
            if (end == -1) end = len
            if (start >= end) {
                start = end + 1; continue
            }
            val equalsIdx = header.indexOf('=', start)
            if (equalsIdx != -1 && equalsIdx < end) {
                val semicolonIdx = header.indexOf(';', equalsIdx)
                val valueEnd = if (semicolonIdx != -1 && semicolonIdx < end) semicolonIdx else end
                val key = header.substring(start, equalsIdx).trim().lowercase()
                val value = header.substring(equalsIdx + 1, valueEnd).trim()
                if (key.isNotEmpty()) map[key] = value
            }
            start = end + 1
        }
        return map
    }

    @JvmStatic
    fun formatBaggage(baggage: Map<String, String>?): String? {
        if (baggage.isNullOrEmpty()) return null

        val sb = StringBuilder(128)
        var isFirst = true

        for ((key, value) in baggage) {
            if (!isFirst) {
                sb.append(',')
            }
            sb.append(key).append('=').append(value)
            isFirst = false
        }

        return sb.toString()
    }
}
