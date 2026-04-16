package com.micronaut.bug.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.http.MediaType
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingRequestWrapper
import org.springframework.web.util.ContentCachingResponseWrapper
import java.nio.charset.StandardCharsets
import java.util.UUID

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
class UnifiedLoggingFilter(objectMapper: ObjectMapper) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    private val prettyMapper: ObjectMapper = objectMapper.copy()
        .enable(SerializationFeature.INDENT_OUTPUT)

    override fun doFilterInternal(
        rq: HttpServletRequest,
        rs: HttpServletResponse,
        chain: FilterChain
    ) {
        val requestId = genTraceId(rq)

        MDC.put(X_REQ_ID, requestId)

        val rqWrapper = ContentCachingRequestWrapper(rq)
        val rsWrapper = ContentCachingResponseWrapper(rs)

        try {
            val isRqMultipart = isMultipart(rq.contentType)
            if (isRqMultipart) {
                logMultipartRequest(rqWrapper)
            } else {
                rqWrapper.inputStream.readAllBytes()
                logSimpleRequest(rqWrapper)
            }

            chain.doFilter(rqWrapper, rsWrapper)
        } finally {
            logResponse(rqWrapper, rsWrapper)
            rsWrapper.copyBodyToResponse()
            MDC.remove(X_REQ_ID)
        }
    }

    private fun genTraceId(rq: HttpServletRequest): String {
        return rq.getHeader(X_REQ_ID)?.takeIf { it.isNotBlank() }
            ?: UUID.randomUUID().toString().replace(DASH, EMPTY)
    }

    private fun logSimpleRequest(rq: ContentCachingRequestWrapper) {
        val content = rq.contentAsByteArray
        val bodyText = if (content.isEmpty()) {
            BODY_NO_CONTENT
        } else {
            formatBody(content, rq.contentType)
        }
        log.info { LOG_TEMPLATE_RQ.format(rq.method, getFullUri(rq), getHeaders(rq), bodyText) }
    }

    private fun logMultipartRequest(rq: HttpServletRequest) {
        val bodyResult = try {
            rq.parts.joinToString(NEW_LINE_DELIMITER) { part ->
                val fileName = part.submittedFileName
                val name = part.name

                if (fileName != null) {
                    PART_PREFIX.format(name, PART_FILE.format(fileName))
                } else {
                    part.inputStream.use { it.readBytes() }.let { bytes ->
                        when {
                            bytes.isEmpty() -> {
                                PART_PREFIX.format(name, PART_EMPTY)
                            }

                            isText(bytes) -> {
                                val content = String(bytes, StandardCharsets.UTF_8)
                                var formatted = formatIfJson(content, part.contentType)
                                if (formatted.length > LIMIT_LOG_SIZE) {
                                    formatted = formatted.take(LIMIT_LOG_SIZE) + TRUNCATED_SUFFIX
                                }
                                PART_PREFIX.format(name, PART_CONTENT.format(formatted))
                            }

                            else -> {
                                PART_PREFIX.format(name, BODY_BINARY)
                            }
                        }
                    }
                }
            }.ifEmpty { BODY_NO_CONTENT }
        } catch (e: Exception) {
            MULTIPART_ERROR_TEMPLATE.format(e.message)
        }

        log.info { LOG_TEMPLATE_RQ.format(rq.method, getFullUri(rq), getHeaders(rq), bodyResult) }
    }

    private fun logResponse(rq: HttpServletRequest, rs: ContentCachingResponseWrapper) {
        val content = rs.contentAsByteArray
        val status = if (rs.status == 0) {
            STATUS_UNKNOWN
        } else {
            rs.status.toString()
        }

        val bodyText = when {
            isMultipart(rs.contentType) -> {
                BODY_MULTIPART_RS
            }

            content.isEmpty() -> {
                BODY_EMPTY
            }

            else -> {
                formatBody(content, rs.contentType)
            }
        }

        log.info { LOG_TEMPLATE_RS.format(getFullUri(rq), status, getResponseHeaders(rs), bodyText) }
    }

    private fun getFullUri(rq: HttpServletRequest): String =
        rq.queryString?.let { URI_WITH_QUERY.format(rq.requestURI, it) } ?: rq.requestURI

    private fun formatBody(content: ByteArray, contentType: String?): String {
        if (!isText(content)) {
            return BODY_BINARY
        }
        return formatIfJson(String(content, StandardCharsets.UTF_8), contentType)
    }

    private fun formatIfJson(body: String?, contentType: String?): String {
        if (body.isNullOrBlank()) {
            return EMPTY
        }
        if (contentType?.contains(MediaType.APPLICATION_JSON_VALUE) == true) {
            return try {
                val tree = prettyMapper.readTree(body)
                prettyMapper.writerWithDefaultPrettyPrinter().writeValueAsString(tree)
            } catch (e: Exception) {
                body
            }
        }
        return body
    }

    private fun getHeaders(rq: HttpServletRequest): String =
        rq.headerNames.asSequence().associateWith { rq.getHeader(it) }.toString()

    private fun getResponseHeaders(rs: HttpServletResponse): String =
        rs.headerNames.associateWith { rs.getHeader(it) }.toString()

    private fun isText(bytes: ByteArray): Boolean =
        bytes.take(TEXT_CHECK_THRESHOLD).none { it == NULL_BYTE }

    private fun isMultipart(contentType: String?): Boolean =
        contentType?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true

    companion object {
        const val X_REQ_ID = "x-req-id"
        private const val EMPTY = ""
        private const val DASH = "-"
        private const val URI_WITH_QUERY = "%s?%s"
        private const val STATUS_UNKNOWN = "UNKNOWN"
        private const val BODY_EMPTY = "[Empty]"
        private const val BODY_BINARY = "Binary data"
        private const val BODY_NO_CONTENT = "No Body"
        private const val BODY_MULTIPART_RS = "[Multipart Response - Raw log disabled]"
        private const val PART_PREFIX = "  [PART] -> Name: %s | %s"
        private const val PART_CONTENT = "Content: %s"
        private const val PART_FILE = "File: %s"
        private const val PART_EMPTY = "Empty content"
        private const val TRUNCATED_SUFFIX = "... [TRUNCATED]"
        private const val MULTIPART_ERROR_TEMPLATE = "[Multipart error: %s]"
        private const val NEW_LINE_DELIMITER = "\n"

        private const val LIMIT_LOG_SIZE = 8192
        private const val TEXT_CHECK_THRESHOLD = 100
        private const val NULL_BYTE: Byte = 0

        private const val LOG_TEMPLATE_RQ = """
================== Service request ==================
URI: %s %s
Headers: %s
Body:
%s
================== /Service request ==================
        """

        private const val LOG_TEMPLATE_RS = """
================== Service response ==================
URI: %s
Status: %s
Headers: %s
Body: %s
================== /Service response ==================
        """
    }
}