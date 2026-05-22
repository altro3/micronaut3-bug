package com.micronaut.bug.trace.http

import com.micronaut.bug.client.HttpClientProperties
import com.micronaut.bug.trace.TraceUtil
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders
import java.net.URI
import java.util.ArrayList

object HttpTraceExtractor {

    private const val QUERY_MARKER = "?"
    private const val PROTOCOL_HTTPS = "https"
    private const val HOST_UNKNOWN = "unknown"
    private const val ATTR_USER_ID = "user.id"
    private const val PREFIX_HTTP_ERROR = "HTTP"

    private const val PORT_HTTPS = 443
    private const val PORT_HTTP = 80
    private const val ERROR_STATUS_THRESHOLD = 400

    fun fillHeadersAttrs(headers: HttpHeaders, prefix: String, isSensitive: Boolean, target: MutableMap<String, Any>) {
        for (name in headers.keys) {
            val lowerName = name.lowercase()
            fillHeadersCore(lowerName, prefix, isSensitive, target) {
                headers[name]
            }
        }
    }

    fun getFullUri(rq: HttpServletRequest): String =
        rq.queryString?.let { "${rq.requestURL}$QUERY_MARKER$it" } ?: rq.requestURL.toString()

    fun fillRequestHeadersAttrs(rq: HttpServletRequest, target: MutableMap<String, Any>) {
        val names = rq.headerNames ?: return
        while (names.hasMoreElements()) {
            val name = names.nextElement()
            val lowerName = name.lowercase()

            fillHeadersCore(lowerName, TraceUtil.PREFIX_HTTP_REQUEST_HEADER, isSensitive = true, target) {
                val enum = rq.getHeaders(name)
                val list = ArrayList<String>(4)
                while (enum.hasMoreElements()) {
                    list.add(enum.nextElement())
                }
                list
            }
        }
    }

    fun fillResponseHeadersAttrs(rs: HttpServletResponse, target: MutableMap<String, Any>) {
        val names = rs.headerNames ?: return
        for (name in names) {
            val lowerName = name.lowercase()

            fillHeadersCore(lowerName, TraceUtil.PREFIX_HTTP_RESPONSE_HEADER, isSensitive = false, target) {
                val headers = rs.getHeaders(name) ?: return@fillHeadersCore null
                val list = ArrayList<String>(headers.size)
                headers.forEach { list.add(it) }
                list
            }
        }
    }

    fun extractServerBaseAttributes(
        methodStr: String,
        scheme: String,
        path: String,
        queryString: String?,
        urlFull: String,
        userAgent: String?,
        remoteAddr: String?,
        serverName: String?,
        serverPort: Int,
        reqBodySize: Long,
        resStatusCode: Int,
        resContentLength: Long?,
        isSlow: Boolean,
        target: MutableMap<String, Any>
    ) {
        extractCommonHttpAttributes(
            host = serverName,
            port = serverPort,
            fallbackPortConfig = null,
            bodySize = reqBodySize,
            statusCode = resStatusCode,
            contentLength = resContentLength,
            isError = resStatusCode >= ERROR_STATUS_THRESHOLD,
            urlFull = urlFull,
            methodStr = methodStr,
            target = target
        )

        target[TraceUtil.ATTR_URL_SCHEME] = scheme
        target[TraceUtil.ATTR_URL_PATH] = path
        if (queryString != null) target[TraceUtil.ATTR_URL_QUERY] = queryString
        if (userAgent != null) target[TraceUtil.ATTR_USER_AGENT_ORIGINAL] = userAgent
        target[TraceUtil.ATTR_CLIENT_ADDRESS] = remoteAddr ?: HOST_UNKNOWN

        if (isSlow) {
            target[TraceUtil.ATTR_HTTP_SLOW_REQUEST] = true
        }
    }

    fun getFullUri(uri: URI): String =
        uri.query?.let { "${uri.path}$QUERY_MARKER$it" } ?: uri.path

    fun extractClientBaseAttributes(
        uri: URI,
        bodySize: Long,
        statusCode: Int?,
        contentLength: Long?,
        isError: Boolean,
        urlFull: String,
        methodStr: String,
        selfServiceName: String,
        serviceNameStr: String,
        userId: String?,
        props: HttpClientProperties,
        target: MutableMap<String, Any>
    ) {
        extractCommonHttpAttributes(
            host = uri.host,
            port = uri.port,
            fallbackPortConfig = props.url.port,
            bodySize = bodySize,
            statusCode = statusCode,
            contentLength = contentLength,
            isError = isError,
            urlFull = urlFull,
            methodStr = methodStr,
            target = target
        )

        if (userId != null) target[ATTR_USER_ID] = userId
        target[TraceUtil.ATTR_CLIENT] = selfServiceName
        target[TraceUtil.ATTR_SERVER] = serviceNameStr
        target[TraceUtil.ATTR_PEER_SERVICE] = props.serviceName ?: (uri.host ?: props.url.host ?: HOST_UNKNOWN)
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun extractCommonHttpAttributes(
        host: String?,
        port: Int,
        fallbackPortConfig: Int?,
        bodySize: Long,
        statusCode: Int?,
        contentLength: Long?,
        isError: Boolean,
        urlFull: String,
        methodStr: String,
        target: MutableMap<String, Any>
    ) {
        target[TraceUtil.ATTR_HTTP_REQUEST_METHOD] = methodStr
        target[TraceUtil.ATTR_URL_FULL] = urlFull
        target[TraceUtil.ATTR_SERVER_ADDRESS] = host ?: HOST_UNKNOWN

        val finalPort = port.takeIf { it != -1 }
            ?: fallbackPortConfig?.takeIf { it != -1 }
            ?: if (urlFull.startsWith(PROTOCOL_HTTPS)) PORT_HTTPS else PORT_HTTP
        target[TraceUtil.ATTR_SERVER_PORT] = finalPort.toLong()

        if (methodStr !in TraceUtil.METHODS_WITHOUT_BODY && bodySize > 0) {
            target[TraceUtil.ATTR_HTTP_REQUEST_BODY_SIZE] = bodySize
        }

        if (statusCode != null) {
            target[TraceUtil.ATTR_HTTP_RESPONSE_STATUS_CODE] = statusCode.toLong()
            if (contentLength != null && contentLength != -1L) {
                target[TraceUtil.ATTR_HTTP_RESPONSE_BODY_SIZE] = contentLength
            }
            if (isError) {
                target[TraceUtil.ATTR_EXCEPTION_MESSAGE] = "$PREFIX_HTTP_ERROR $statusCode"
            }
        }
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun fillHeadersCore(
        lowerName: String,
        prefix: String,
        isSensitive: Boolean,
        target: MutableMap<String, Any>,
        crossinline valuesProvider: () -> List<String>?
    ) {
        if (TraceUtil.OTEL_MAPPED_HEADERS.contains(lowerName)) return

        val key = "$prefix$lowerName"

        if (isSensitive && TraceUtil.SENSITIVE_HEADERS.contains(lowerName)) {
            target[key] = TraceUtil.MASKED_VALUES
            return
        }

        val values = valuesProvider() ?: return
        val size = values.size
        if (size == 0) return

        val list = ArrayList<String>(size)
        for (j in 0 until size) {
            list.add(values[j])
        }
        target[key] = list
    }
}
