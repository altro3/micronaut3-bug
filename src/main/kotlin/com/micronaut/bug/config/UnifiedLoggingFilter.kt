package com.micronaut.bug.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.ReadListener
import jakarta.servlet.ServletInputStream
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletRequestWrapper
import jakarta.servlet.http.HttpServletResponse
import jakarta.servlet.http.Part
import org.slf4j.MDC
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingResponseWrapper
import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.InputStreamReader
import java.util.UUID

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
class UnifiedLoggingFilter(
    objectMapper: ObjectMapper,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    private val prettyMapper: ObjectMapper = objectMapper.copy()
        .enable(SerializationFeature.INDENT_OUTPUT)

    override fun doFilterInternal(
        rq: HttpServletRequest,
        rs: HttpServletResponse,
        chain: FilterChain,
    ) {
        val rqId = rq.getHeader(X_REQ_ID)?.takeIf { it.isNotBlank() } ?: genTraceId()
        MDC.put(X_REQ_ID, rqId)

        if (!log.isDebugEnabled()) {
            chain.doFilter(rq, rs)
            return
        }

        val rsWrapper = ContentCachingResponseWrapper(rs)
        val currentRq = wrapAndLogRequest(rq)

        try {
            chain.doFilter(currentRq, rsWrapper)
        } finally {
            logResponse(currentRq, rsWrapper)
            rsWrapper.copyBodyToResponse()
            MDC.remove(X_REQ_ID)
        }
    }

    private fun wrapAndLogRequest(rq: HttpServletRequest): HttpServletRequest {
        val isMultipart = isMultipart(rq.contentType)
        val (wrapper, bodyResult) = if (isMultipart) {
            val mw = MultipartTypeWrapper(rq)
            mw to getMultipartBody(mw)
        } else {
            val bytes = rq.inputStream.readAllBytes()
            val cw = CachedBodyRequestWrapper(rq, bytes)
            val body = if (bytes.isEmpty()) BODY_EMPTY else formatBody(bytes, rq.contentType)
            cw to body
        }

        log.debug {
            """
================== Service request ==================
URI: ${rq.method} ${getFullUri(rq)}
Headers: ${getHeaders(rq)}
Body:
$bodyResult
================== /Service request ==================
            """.trimIndent()
        }
        return wrapper
    }

    private fun getMultipartBody(rq: MultipartTypeWrapper): String =
        runCatching {
            rq.parts.joinToString(NEW_LINE_DELIMITER) { part ->
                val bytes = (part as ObservedPart).getContentBytes()
                val description = part.submittedFileName?.let {
                    PART_FILE_INFO.format(part.name, it, part.size)
                } ?: part.name

                processPartContent(bytes, part.contentType, description)
            }.ifEmpty { BODY_EMPTY }
        }.getOrElse { "[MULTIPART ERROR: ${it.message}]" }

    private fun processPartContent(bytes: ByteArray, ct: String?, description: String): String =
        when {
            bytes.isEmpty() -> PART_PREFIX.format(description, PART_EMPTY)

            isText(bytes) -> {
                val formatted = formatIfJson(bytes, ct).let {
                    if (it.length > LIMIT_LOG_SIZE) {
                        it.take(LIMIT_LOG_SIZE) + TRUNCATED_SUFFIX
                    } else {
                        it
                    }
                }
                PART_PREFIX.format(description, "Content: $formatted")
            }

            else -> PART_PREFIX.format(description, BODY_BINARY)
        }

    private fun genTraceId(): String =
        UUID.randomUUID().toString().replace(DASH, EMPTY)

    private fun logResponse(rq: HttpServletRequest, rs: ContentCachingResponseWrapper) {
        val statusInt = rs.status.takeIf { it != 0 } ?: 200

        // Безопасно ищем описание статуса
        val statusMessage = runCatching {
            HttpStatus.resolve(statusInt)?.reasonPhrase
        }.getOrNull() ?: EMPTY // Если статус кастомный (resolve вернет null), будет просто пустая строка

        val contentType = rs.contentType
        val bodyText = when {
            isMultipart(contentType) -> formatMultipartResponse(rs.contentAsByteArray, contentType)

            rs.contentAsByteArray.isEmpty() -> BODY_EMPTY

            else -> formatBody(rs.contentAsByteArray, contentType)
        }

        log.debug {
            """
================== Service response ==================
URI: ${rq.method} ${getFullUri(rq)}
Status: $statusInt $statusMessage
Headers: ${getResponseHeaders(rs)}
Body:
$bodyText
================== /Service response ==================
            """.trimIndent()
        }
    }

    private fun formatMultipartResponse(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) {
            return BODY_EMPTY
        }

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

                    val contentLines = lines.dropWhile { it.isNotBlank() }
                    val content = contentLines
                        .drop(1)
                        .joinToString(NEW_LINE_DELIMITER)

                    val name = headers.find { it.contains(NAME_MARKER) }
                        ?.substringAfter(NAME_EQUALS)
                        ?.substringBefore(SEMICOLON)
                        ?.replace(QUOTE, EMPTY) ?: UNKNOWN_PART

                    val fileName = headers.find { it.contains(FILENAME_MARKER) }
                        ?.substringAfter(FILENAME_EQUALS)
                        ?.substringBefore(QUOTE)

                    val partCt = headers.find { it.contains(CONTENT_TYPE_HEADER) }
                        ?.substringAfter(COLON_SPACE)

                    val contentBytes = content.toByteArray()
                    val description = if (fileName != null) {
                        PART_FILE_INFO.format(name, fileName, contentBytes.size)
                    } else {
                        name
                    }

                    if (fileName != null && !isAlwaysTextField(partCt)) {
                        if (isText(contentBytes)) {
                            processPartContent(contentBytes, partCt, description)
                        } else {
                            PART_PREFIX.format(description, BODY_BINARY)
                        }
                    } else {
                        processPartContent(contentBytes, partCt, description)
                    }
                }
        }.getOrElse { BODY_MULTIPART_RS }
    }

    private fun getFullUri(rq: HttpServletRequest): String =
        rq.queryString?.let { "${rq.requestURI}?$it" } ?: rq.requestURI

    private fun formatBody(content: ByteArray, contentType: String?): String {
        if (!isText(content)) {
            return BODY_BINARY
        }
        return formatIfJson(content, contentType)
    }

    private fun formatIfJson(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) {
            return EMPTY
        }

        val isJson = contentType?.contains(MediaType.APPLICATION_JSON_VALUE) == true
                || isJsonContent(bytes)

        return if (isJson) {
            runCatching {
                val tree = prettyMapper.readTree(bytes)
                prettyMapper.writeValueAsString(tree)
            }.getOrElse { String(bytes) }
        } else {
            String(bytes)
        }
    }

    private fun getHeaders(rq: HttpServletRequest): String =
        rq.headerNames.asSequence().associateWith { rq.getHeader(it) }.toString()

    private fun getResponseHeaders(rs: HttpServletResponse): String =
        rs.headerNames.associateWith { rs.getHeader(it) }.toString()

    private fun isMultipart(contentType: String?): Boolean =
        contentType?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true

    private fun isAlwaysTextField(contentType: String?): Boolean {
        if (contentType == null) {
            return false
        }
        val ct = contentType.lowercase()
        return ct.contains(MediaType.APPLICATION_JSON_VALUE)
                || ct.contains(TEXT_CT_MARKER)
    }

    private class CachedBodyRequestWrapper(
        rq: HttpServletRequest,
        private val body: ByteArray,
    ) : HttpServletRequestWrapper(rq) {

        override fun getInputStream(): ServletInputStream {
            val byteArrayInputStream = ByteArrayInputStream(body)
            return object : ServletInputStream() {

                override fun read(): Int = byteArrayInputStream.read()

                override fun isFinished(): Boolean = byteArrayInputStream.available() == 0

                override fun isReady(): Boolean = true

                override fun setReadListener(readListener: ReadListener?) {}
            }
        }

        override fun getReader(): BufferedReader =
            BufferedReader(InputStreamReader(getInputStream()))
    }

    private class MultipartTypeWrapper(
        rq: HttpServletRequest,
    ) : HttpServletRequestWrapper(rq) {

        private val cachedParts by lazy {
            super.getParts().map { part ->
                val bytes = part.inputStream.use { it.readBytes() }
                val ct = part.contentType

                val isJson = (ct == null || ct == MediaType.APPLICATION_OCTET_STREAM_VALUE) && isJsonContent(bytes)
                val finalCt = if (isJson) {
                    MediaType.APPLICATION_JSON_VALUE
                } else {
                    ct
                }

                ObservedPart(part, finalCt, bytes)
            }
        }

        override fun getParts(): Collection<Part> = cachedParts

        override fun getPart(name: String): Part? = cachedParts.find { it.name == name }
    }

    private class ObservedPart(
        private val original: Part,
        private val overriddenContentType: String?,
        private val bytes: ByteArray,
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

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_MULTIPART_RS = "[MULTIPART RAW DISABLED]"

        private const val PART_PREFIX = "  [PART] -> Name: %s | %s"
        private const val PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"
        private const val PART_EMPTY = "[EMPTY CONTENT]"
        private const val TRUNCATED_SUFFIX = "... [TRUNCATED]"

        private const val NEW_LINE_DELIMITER = "\n"
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

        private const val JSON_BYTE_OBJECT = '{'.code.toByte()
        private const val JSON_BYTE_ARRAY = '['.code.toByte()

        private const val LIMIT_LOG_SIZE = 8192
        private const val TEXT_CHECK_THRESHOLD = 100
        private const val NULL_BYTE: Byte = 0

        private fun isJsonContent(bytes: ByteArray): Boolean {
            val firstByte = bytes.find { it > 32 }
            return firstByte == JSON_BYTE_OBJECT || firstByte == JSON_BYTE_ARRAY
        }

        private fun isText(bytes: ByteArray): Boolean =
            bytes.take(TEXT_CHECK_THRESHOLD).none { it == NULL_BYTE }
    }
}
