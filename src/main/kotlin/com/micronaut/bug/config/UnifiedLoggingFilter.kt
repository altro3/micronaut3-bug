package com.micronaut.bug.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletRequestWrapper
import jakarta.servlet.http.HttpServletResponse
import jakarta.servlet.http.Part
import org.slf4j.MDC
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.http.MediaType
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingRequestWrapper
import org.springframework.web.util.ContentCachingResponseWrapper
import java.util.UUID

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
class UnifiedLoggingFilter(objectMapper: ObjectMapper) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}
    private val prettyMapper: ObjectMapper = objectMapper.copy()
        .enable(SerializationFeature.INDENT_OUTPUT)

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        val requestId = rq.getHeader(X_REQ_ID)?.takeIf { it.isNotBlank() } ?: genTraceId()
        MDC.put(X_REQ_ID, requestId)

        val rsWrapper = ContentCachingResponseWrapper(rs)
        val contentType = rq.contentType
        val isRqMultipart = isMultipart(contentType)

        val currentRq = if (isRqMultipart) {
            MultipartTypeWrapper(rq).also { logMultipartRequest(it) }
        } else {
            ContentCachingRequestWrapper(rq).apply {
                inputStream.readAllBytes()
                logSimpleRequest(this)
            }
        }

        try {
            chain.doFilter(currentRq, rsWrapper)
        } finally {
            logResponse(currentRq, rsWrapper)
            rsWrapper.copyBodyToResponse()
            MDC.remove(X_REQ_ID)
        }
    }

    private fun logMultipartRequest(rq: HttpServletRequest) {
        val bodyResult = runCatching {
            rq.parts.joinToString(NEW_LINE_DELIMITER) { part ->
                val fileName = part.submittedFileName
                val name = part.name
                val ct = part.contentType

                val description = if (fileName != null) {
                    PART_FILE_INFO.format(name, fileName)
                } else {
                    name
                }

                // УБРАЛИ лишнюю проверку fileName, теперь всегда читаем байты
                val bytes = if (part is ObservedPart) {
                    part.getContentBytes()
                } else {
                    part.inputStream.use { it.readBytes() }
                }

                processPartContent(bytes, ct, description)
            }.ifEmpty { BODY_NO_CONTENT }
        }.getOrElse { MULTIPART_ERROR_TEMPLATE.format(it.message) }

        log.info { LOG_TEMPLATE_RQ.format(rq.method, getFullUri(rq), getHeaders(rq), bodyResult) }
    }

    private fun processPartContent(bytes: ByteArray, ct: String?, description: String): String = when {
        bytes.isEmpty() -> PART_PREFIX.format(description, PART_EMPTY)
        isText(bytes) -> {
            val content = String(bytes)
            val formatted = formatIfJson(content, ct).let {
                if (it.length > LIMIT_LOG_SIZE) it.take(LIMIT_LOG_SIZE) + TRUNCATED_SUFFIX else it
            }
            PART_PREFIX.format(description, PART_CONTENT.format(formatted))
        }

        else -> PART_PREFIX.format(description, BODY_BINARY)
    }

    private fun genTraceId(): String =
        UUID.randomUUID().toString().replace(DASH, EMPTY)

    private fun logSimpleRequest(rq: ContentCachingRequestWrapper) {
        val bodyText = if (rq.contentAsByteArray.isEmpty()) BODY_NO_CONTENT else formatBody(rq.contentAsByteArray, rq.contentType)
        log.info { LOG_TEMPLATE_RQ.format(rq.method, getFullUri(rq), getHeaders(rq), bodyText) }
    }

    private fun logResponse(rq: HttpServletRequest, rs: ContentCachingResponseWrapper) {
        val status = rs.status.takeIf { it != 0 }?.toString() ?: STATUS_UNKNOWN
        val contentType = rs.contentType

        val bodyText = when {
            isMultipart(contentType) -> formatMultipartResponse(rs.contentAsByteArray, contentType)
            rs.contentAsByteArray.isEmpty() -> BODY_EMPTY
            else -> formatBody(rs.contentAsByteArray, contentType)
        }

        log.info { LOG_TEMPLATE_RS.format(getFullUri(rq), status, getResponseHeaders(rs), bodyText) }
    }

    private fun formatMultipartResponse(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) return BODY_EMPTY

        val boundary = contentType?.split(BOUNDARY_MARKER)?.getOrNull(1)?.let {
            DASH_PREFIX + it
        } ?: return BODY_MULTIPART_RS

        return runCatching {
            val bodyString = String(bytes)
            bodyString.split(boundary)
                .filter { it.contains(CONTENT_DISPOSITION_MARKER) }
                .joinToString(NEW_LINE_DELIMITER) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headers = lines.takeWhile { it.isNotBlank() }
                    val content = lines.dropWhile { it.isNotBlank() }.drop(1).joinToString(NEW_LINE_DELIMITER)

                    val name = headers.find { it.contains(NAME_MARKER) }
                        ?.substringAfter(NAME_EQUALS)?.substringBefore(SEMICOLON)?.replace(QUOTE, EMPTY) ?: UNKNOWN_PART
                    val fileName = headers.find { it.contains(FILENAME_MARKER) }
                        ?.substringAfter(FILENAME_EQUALS)?.substringBefore(QUOTE)
                    val partCt = headers.find { it.contains(CONTENT_TYPE_HEADER) }
                        ?.substringAfter(COLON_SPACE)

                    val description = if (fileName != null) PART_FILE_INFO.format(name, fileName) else name

                    // ИСПРАВЛЕНО: Теперь всегда вызываем processPartContent для консистентности маркеров
                    if (fileName != null && !isAlwaysTextField(partCt)) {
                        val contentBytes = content.toByteArray()
                        if (isText(contentBytes)) {
                            processPartContent(contentBytes, partCt, description)
                        } else {
                            PART_PREFIX.format(description, BODY_BINARY)
                        }
                    } else {
                        processPartContent(content.toByteArray(), partCt, description)
                    }
                }
        }.getOrElse { BODY_MULTIPART_RS }
    }

    private fun getFullUri(rq: HttpServletRequest): String =
        rq.queryString?.let { URI_WITH_QUERY.format(rq.requestURI, it) } ?: rq.requestURI

    private fun formatBody(content: ByteArray, contentType: String?): String =
        if (!isText(content)) BODY_BINARY else formatIfJson(String(content), contentType)

    private fun formatIfJson(body: String?, contentType: String?): String {
        if (body.isNullOrBlank()) return EMPTY

        val trimmed = body.trim()
        val isJson = contentType?.contains(MediaType.APPLICATION_JSON_VALUE) == true
                || trimmed.startsWith(JSON_OBJECT_START)
                || trimmed.startsWith(JSON_ARRAY_START)

        return if (isJson) {
            runCatching {
                val tree = prettyMapper.readTree(trimmed)
                prettyMapper.writerWithDefaultPrettyPrinter().writeValueAsString(tree)
            }.getOrDefault(body)
        } else {
            body
        }
    }

    private fun getHeaders(rq: HttpServletRequest): String =
        rq.headerNames.asSequence().associateWith { rq.getHeader(it) }.toString()

    private fun getResponseHeaders(rs: HttpServletResponse): String =
        rs.headerNames.associateWith { rs.getHeader(it) }.toString()

    private fun isText(bytes: ByteArray): Boolean =
        bytes.take(TEXT_CHECK_THRESHOLD).none { it == NULL_BYTE }

    private fun isMultipart(contentType: String?): Boolean =
        contentType?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true

    private fun isAlwaysTextField(contentType: String?): Boolean {
        if (contentType == null) return false
        val ct = contentType.lowercase()
        return ct.contains(MediaType.APPLICATION_JSON_VALUE) || ct.contains(TEXT_CT_MARKER)
    }

    private class MultipartTypeWrapper(request: HttpServletRequest) : HttpServletRequestWrapper(request) {

        private val cachedParts by lazy {
            super.getParts().map { part ->
                val bytes = part.inputStream.use { it.readBytes() }
                val ct = part.contentType

                val isJson = (ct == null || ct == MediaType.APPLICATION_OCTET_STREAM_VALUE) && isJsonContent(bytes)
                val finalCt = if (isJson) MediaType.APPLICATION_JSON_VALUE else ct

                ObservedPart(part, finalCt, bytes)
            }
        }

        override fun getParts(): Collection<Part> = cachedParts

        override fun getPart(name: String): Part? = cachedParts.find { it.name == name }

        companion object {

            private fun isJsonContent(bytes: ByteArray): Boolean {
                val s = String(bytes).trim()
                return s.startsWith(JSON_OBJECT_START) || s.startsWith(JSON_ARRAY_START)
            }
        }
    }

    private class ObservedPart(
        private val original: Part,
        private val overriddenContentType: String?,
        private val bytes: ByteArray
    ) : Part by original {

        override fun getContentType() = overriddenContentType

        override fun getInputStream() = bytes.inputStream()

        override fun getSize() = bytes.size.toLong()

        fun getContentBytes() = bytes
    }

    companion object {
        const val X_REQ_ID = "x-req-id"
        private const val EMPTY = ""
        private const val DASH = "-"
        private const val URI_WITH_QUERY = "%s?%s"
        private const val STATUS_UNKNOWN = "UNKNOWN"

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_NO_CONTENT = "[NO CONTENT]"
        private const val BODY_MULTIPART_RS = "[MULTIPART RAW DISABLED]"

        private const val PART_PREFIX = "  [PART] -> Name: %s | %s"
        private const val PART_CONTENT = "Content: %s"
        private const val PART_FILE = "File: %s"
        private const val PART_FILE_INFO = "%s (File: %s)"
        private const val PART_EMPTY = "[EMPTY CONTENT]"
        private const val TRUNCATED_SUFFIX = "... [TRUNCATED]"
        private const val MULTIPART_ERROR_TEMPLATE = "[MULTIPART ERROR: %s]"

        private const val NEW_LINE_DELIMITER = "\n"
        private const val JSON_OBJECT_START = "{"
        private const val JSON_ARRAY_START = "["
        private const val TEXT_CT_MARKER = "text"
        private const val BOUNDARY_MARKER = "boundary="
        private const val DASH_PREFIX = "--"
        private const val CONTENT_DISPOSITION_MARKER = "Content-Disposition"
        private const val NAME_MARKER = "name="
        private const val NAME_EQUALS = "name=\""
        private const val FILENAME_MARKER = "filename="
        private const val FILENAME_EQUALS = "filename=\""
        private const val CONTENT_TYPE_HEADER = "Content-Type"
        private const val COLON_SPACE = ": "
        private const val SEMICOLON = ";"
        private const val QUOTE = "\""
        private const val UNKNOWN_PART = "UNKNOWN"

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
