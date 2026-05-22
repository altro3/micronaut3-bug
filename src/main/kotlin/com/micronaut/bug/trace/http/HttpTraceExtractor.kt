package com.micronaut.bug.trace.http

import com.micronaut.bug.client.HttpClientProperties
import com.micronaut.bug.trace.TraceUtil
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpResponse

object HttpTraceExtractor {

    private const val QUERY_MARKER = "?"
    private const val PROTOCOL_HTTPS = "https"
    private const val HOST_UNKNOWN = "unknown"
    private const val ATTR_USER_ID = "user.id"

    // ==========================================================================================
    // 1. УНИВЕРСАЛЬНЫЙ СЛОЙ SPRING (HttpHeaders подходит для WebFlux, Netty и RestTemplate)
    // ==========================================================================================

    fun fillHeadersAttrs(headers: HttpHeaders, prefix: String, isSensitive: Boolean, target: MutableMap<String, Any>) {
        for (name in headers.keys) {
            val lowerName = name.lowercase()

            fillHeadersCore(lowerName, prefix, isSensitive, target) {
                headers[name]
            }
        }
    }

    // ==========================================================================================
    // 2. СИНХРОННЫЙ СЕРВЛЕТНЫЙ СЛОЙ (Tomcat / Jetty / Jakarta Servlet API)
    // ==========================================================================================

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

    fun extractBaseAttributes(rq: HttpServletRequest, rs: HttpServletResponse, attrs: HashMap<String, Any>, isSlow: Boolean) {
        attrs[TraceUtil.ATTR_URL_FULL] = getFullUri(rq)
        attrs[TraceUtil.ATTR_URL_SCHEME] = rq.scheme
        attrs[TraceUtil.ATTR_URL_PATH] = rq.requestURI
        rq.queryString?.let { attrs[TraceUtil.ATTR_URL_QUERY] = it }

        attrs[TraceUtil.ATTR_HTTP_REQUEST_METHOD] = rq.method
        rq.getHeader(HttpHeaders.USER_AGENT)?.let { attrs[TraceUtil.ATTR_USER_AGENT_ORIGINAL] = it }

        if (rq.method !in TraceUtil.METHODS_WITHOUT_BODY) {
            val reqSize = rq.contentLength.toLong()
            if (reqSize != -1L) attrs[TraceUtil.ATTR_HTTP_REQUEST_BODY_SIZE] = reqSize
        }

        if (isSlow) attrs[TraceUtil.ATTR_HTTP_SLOW_REQUEST] = true

        rs.getHeader(HttpHeaders.CONTENT_LENGTH)?.toLongOrNull()?.let { attrs[TraceUtil.ATTR_HTTP_RESPONSE_BODY_SIZE] = it }

        attrs[TraceUtil.ATTR_SERVER_ADDRESS] = rq.serverName
        attrs[TraceUtil.ATTR_SERVER_PORT] = rq.serverPort.toLong()
        attrs[TraceUtil.ATTR_CLIENT_ADDRESS] = rq.remoteAddr
        attrs[TraceUtil.ATTR_HTTP_RESPONSE_STATUS_CODE] = rs.status.toLong()
    }

    // ==========================================================================================
    // 3. БАЗОВЫЕ МЕТАДАННЫЕ ИСХОДЯЩЕГО КЛИЕНТА (Spring Client API)
    // ==========================================================================================

    fun getFullUri(rq: HttpRequest): String =
        rq.uri.query?.let { "${rq.uri.path}$QUERY_MARKER$it" } ?: rq.uri.path

    fun extractBaseAttributes(
        rq: HttpRequest, body: ByteArray, rs: ClientHttpResponse?, urlFull: String, methodStr: String,
        selfServiceName: String, serviceNameStr: String, userId: String?, props: HttpClientProperties,
        target: HashMap<String, Any>
    ) {
        if (userId != null) target[ATTR_USER_ID] = userId
        target[TraceUtil.ATTR_HTTP_REQUEST_METHOD] = methodStr
        target[TraceUtil.ATTR_URL_FULL] = urlFull

        val host = rq.uri.host ?: props.url.host ?: HOST_UNKNOWN
        target[TraceUtil.ATTR_SERVER_ADDRESS] = host

        val port = rq.uri.port.takeIf { it != -1 } ?: props.url.port.takeIf { it != -1 } ?: if (urlFull.startsWith(PROTOCOL_HTTPS)) 443 else 80
        target[TraceUtil.ATTR_SERVER_PORT] = port.toLong()

        target[TraceUtil.ATTR_CLIENT] = selfServiceName
        target[TraceUtil.ATTR_SERVER] = serviceNameStr
        target[TraceUtil.ATTR_PEER_SERVICE] = props.serviceName ?: host

        if (methodStr !in TraceUtil.METHODS_WITHOUT_BODY && body.isNotEmpty()) {
            target[TraceUtil.ATTR_HTTP_REQUEST_BODY_SIZE] = body.size.toLong()
        }

        if (rs != null) {
            val statusCode = rs.statusCode
            target[TraceUtil.ATTR_HTTP_RESPONSE_STATUS_CODE] = statusCode.value().toLong()
            rs.headers.contentLength.takeIf { it != -1L }?.let { target[TraceUtil.ATTR_HTTP_RESPONSE_BODY_SIZE] = it }
            if (statusCode.isError) target[TraceUtil.ATTR_EXCEPTION_MESSAGE] = "HTTP ${statusCode.value()}"
        }
    }

    // ==========================================================================================
    // 4. СВЕРХБЫСТРОЕ ОБЩЕЕ ЯДРО ОБРАБОТКИ ХЭДЕРОВ (Единая инлайновая логика)
    // ==========================================================================================

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
