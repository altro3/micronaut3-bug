package com.micronaut.bug.trace

import com.micronaut.bug.trace.TraceUtil.ATTR_CLIENT_ADDRESS
import com.micronaut.bug.trace.TraceUtil.ATTR_HTTP_REQUEST_BODY_SIZE
import com.micronaut.bug.trace.TraceUtil.ATTR_HTTP_REQUEST_METHOD
import com.micronaut.bug.trace.TraceUtil.ATTR_HTTP_RESPONSE_BODY_SIZE
import com.micronaut.bug.trace.TraceUtil.ATTR_HTTP_RESPONSE_STATUS_CODE
import com.micronaut.bug.trace.TraceUtil.ATTR_HTTP_SLOW_REQUEST
import com.micronaut.bug.trace.TraceUtil.ATTR_SERVER_ADDRESS
import com.micronaut.bug.trace.TraceUtil.ATTR_SERVER_PORT
import com.micronaut.bug.trace.TraceUtil.ATTR_URL_FULL
import com.micronaut.bug.trace.TraceUtil.ATTR_URL_PATH
import com.micronaut.bug.trace.TraceUtil.ATTR_URL_QUERY
import com.micronaut.bug.trace.TraceUtil.ATTR_URL_SCHEME
import com.micronaut.bug.trace.TraceUtil.ATTR_USER_AGENT_ORIGINAL
import com.micronaut.bug.trace.TraceUtil.MASKED_VALUES
import com.micronaut.bug.trace.TraceUtil.METHODS_WITHOUT_BODY
import com.micronaut.bug.trace.TraceUtil.OTEL_MAPPED_HEADERS
import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.TraceUtil.SENSITIVE_HEADERS
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders.CONTENT_LENGTH
import org.springframework.http.HttpHeaders.USER_AGENT

object HttpTraceExtractor {

    private const val QUERY_MARKER = "?"

    fun getFullUri(rq: HttpServletRequest): String {
        val q = rq.queryString ?: return rq.requestURL.toString()
        val sb = StringBuilder(128)
        sb.append(rq.requestURL).append(QUERY_MARKER).append(q)
        return sb.toString()
    }

    fun fillRequestHeadersAttrs(rq: HttpServletRequest, target: MutableMap<String, Any>) {
        val names = rq.headerNames ?: return
        while (names.hasMoreElements()) {
            val name = names.nextElement()
            val lowerName = name.lowercase()

            if (OTEL_MAPPED_HEADERS.contains(lowerName)) continue

            val key = "$PREFIX_HTTP_REQUEST_HEADER$lowerName"

            if (SENSITIVE_HEADERS.contains(lowerName)) {
                target[key] = MASKED_VALUES
            } else {
                val headersEnum = rq.getHeaders(name)
                if (!headersEnum.hasMoreElements()) continue

                val first = headersEnum.nextElement()
                if (!headersEnum.hasMoreElements()) {
                    target[key] = first
                } else {
                    val list = ArrayList<String>(4)
                    list.add(first)
                    while (headersEnum.hasMoreElements()) {
                        list.add(headersEnum.nextElement())
                    }
                    target[key] = list
                }
            }
        }
    }

    fun fillResponseHeadersAttrs(rs: HttpServletResponse, target: MutableMap<String, Any>) {
        val names = rs.headerNames ?: return

        for (name in names) {
            val lowerName = name.lowercase()

            if (OTEL_MAPPED_HEADERS.contains(lowerName)) continue

            val key = "$PREFIX_HTTP_RESPONSE_HEADER$lowerName"
            val headers = rs.getHeaders(name) ?: continue

            val list = ArrayList<String>(4)
            headers.forEach { value ->
                list.add(value)
            }

            if (list.isNotEmpty()) {
                target[key] = list
            }
        }
    }

    fun extractBaseAttributes(rq: HttpServletRequest, rs: HttpServletResponse, attrs: HashMap<String, Any>, isSlow: Boolean) {
        attrs[ATTR_URL_FULL] = getFullUri(rq)
        attrs[ATTR_URL_SCHEME] = rq.scheme
        attrs[ATTR_URL_PATH] = rq.requestURI
        rq.queryString?.let { attrs[ATTR_URL_QUERY] = it }

        attrs[ATTR_HTTP_REQUEST_METHOD] = rq.method
        rq.getHeader(USER_AGENT)?.let { attrs[ATTR_USER_AGENT_ORIGINAL] = it }

        if (rq.method !in METHODS_WITHOUT_BODY) {
            val reqSize = rq.contentLength.toLong()
            if (reqSize != -1L) {
                attrs[ATTR_HTTP_REQUEST_BODY_SIZE] = reqSize
            }
        }

        if (isSlow) {
            attrs[ATTR_HTTP_SLOW_REQUEST] = true
        }

        rs.getHeader(CONTENT_LENGTH)
            ?.toLongOrNull()
            ?.let { attrs[ATTR_HTTP_RESPONSE_BODY_SIZE] = it }

        attrs[ATTR_SERVER_ADDRESS] = rq.serverName
        attrs[ATTR_SERVER_PORT] = rq.serverPort.toLong()
        attrs[ATTR_CLIENT_ADDRESS] = rq.remoteAddr
        attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = rs.status.toLong()
    }
}
