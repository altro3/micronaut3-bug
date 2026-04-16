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
        val requestId = rq.getHeader(X_REQ_ID)?.takeIf { it.isNotBlank() } ?: genTraceId()
        MDC.put(X_REQ_ID, requestId)

        try {
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
            }
        } finally {
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
            rq.parts.joinToString(STRING_NEW_LINE) { part ->
                val bytes = (part as ObservedPart).getContentBytes()
                val description = part.submittedFileName?.let {
                    TEMPLATE_PART_FILE_INFO.format(part.name, it, part.size)
                } ?: part.name

                processPartContent(bytes, part.contentType, description)
            }.ifEmpty { BODY_EMPTY }
        }.getOrElse { "[MULTIPART ERROR: ${it.message}]" }

    private fun processPartContent(bytes: ByteArray, ct: String?, description: String): String =
        when {
            bytes.isEmpty() -> TEMPLATE_PART_PREFIX.format(description, BODY_EMPTY)

            isText(bytes) -> {
                val formatted = formatIfJson(bytes, ct).let {
                    if (it.length > LIMIT_LOG_SIZE) {
                        it.take(LIMIT_LOG_SIZE) + SUFFIX_TRUNCATED
                    } else {
                        it
                    }
                }
                TEMPLATE_PART_PREFIX.format(description, "Content: $formatted")
            }

            else -> TEMPLATE_PART_PREFIX.format(description, BODY_BINARY)
        }

    private fun logResponse(rq: HttpServletRequest, rs: ContentCachingResponseWrapper) {
        val statusInt = rs.status.takeIf { it != 0 } ?: 200

        // Безопасно ищем описание статуса
        val statusMessage = runCatching {
            HttpStatus.resolve(statusInt)?.reasonPhrase
        }.getOrNull() ?: STRING_EMPTY // Если статус кастомный (resolve вернет null), будет просто пустая строка

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

        val boundary = contentType?.split(MARKER_BOUNDARY)?.getOrNull(1)?.let {
            PREFIX_DASH + it
        } ?: return BODY_MULTIPART_RS

        return runCatching {
            String(bytes).split(boundary)
                .filter { it.contains(MARKER_CONTENT_DISPOSITION) }
                .joinToString(STRING_NEW_LINE) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headers = lines.takeWhile { it.isNotBlank() }

                    val content = lines.dropWhile { it.isNotBlank() }
                        .drop(1)
                        .joinToString(STRING_NEW_LINE)

                    var name: String? = null
                    var fileName: String? = null
                    var partCt: String? = null

                    headers.forEach { header ->
                        when {
                            header.contains(MARKER_NAME) -> {
                                name = header.substringAfter(EQUALS_NAME)
                                    .substringBefore(STRING_SEMICOLON)
                                    .substringBefore(STRING_QUOTE)
                            }

                            header.contains(MARKER_FILENAME) -> {
                                fileName = header.substringAfter(EQUALS_FILENAME)
                                    .substringBefore(STRING_QUOTE)
                            }

                            header.contains(HEADER_CONTENT_TYPE) -> {
                                partCt = header.substringAfter(STRING_COLON_SPACE)
                            }
                        }
                    }

                    val finalName = name ?: PART_UNKNOWN
                    val contentBytes = content.toByteArray()
                    val description = if (fileName != null) {
                        TEMPLATE_PART_FILE_INFO.format(finalName, fileName, contentBytes.size.toLong())
                    } else {
                        finalName
                    }

                    val isBinaryFile = fileName != null && !isAlwaysTextField(partCt) && !isText(contentBytes)
                    if (isBinaryFile) {
                        TEMPLATE_PART_PREFIX.format(description, BODY_BINARY)
                    } else {
                        processPartContent(contentBytes, partCt, description)
                    }
                }
        }.getOrElse { BODY_MULTIPART_RS }
    }

    private fun formatBody(content: ByteArray, contentType: String?): String {
        if (!isText(content)) {
            return BODY_BINARY
        }
        return formatIfJson(content, contentType)
    }

    private fun formatIfJson(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) {
            return STRING_EMPTY
        }

        val isJson = contentType?.contains(MediaType.APPLICATION_JSON_VALUE) == true
                || isJsonContent(bytes)

        return if (isJson) {
            runCatching {
                prettyMapper.writeValueAsString(prettyMapper.readTree(bytes))
            }.getOrElse { String(bytes) }
        } else {
            String(bytes)
        }
    }

    private fun getHeaders(rq: HttpServletRequest): String =
        rq.headerNames.asSequence().associateWith { rq.getHeader(it) }.toString()

    private fun getResponseHeaders(rs: HttpServletResponse): String =
        rs.headerNames.associateWith { rs.getHeader(it) }.toString()

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

        private const val STRING_EMPTY = ""
        private const val STRING_DASH = "-"
        private const val STRING_QUOTE = "\""
        private const val STRING_NEW_LINE = "\n"
        private const val STRING_COLON_SPACE = ": "
        private const val STRING_SEMICOLON = ";"

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_MULTIPART_RS = "[MULTIPART RAW DISABLED]"

        private const val PART_UNKNOWN = "UNKNOWN"

        private const val TEMPLATE_PART_PREFIX = "  [PART] -> Name: %s | %s"
        private const val TEMPLATE_PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"

        private const val SUFFIX_TRUNCATED = "... [TRUNCATED]"

        private const val MARKER_BOUNDARY = "boundary="
        private const val MARKER_CONTENT_DISPOSITION = "Content-Disposition"
        private const val MARKER_NAME = "name="
        private const val MARKER_FILENAME = "filename="
        private const val MARKER_TEXT = "text"

        private const val EQUALS_NAME = "name=\""
        private const val EQUALS_FILENAME = "filename=\""

        private const val PREFIX_DASH = "--"
        private const val HEADER_CONTENT_TYPE = "Content-Type"

        private const val LIMIT_LOG_SIZE = 8192
        private const val LIMIT_TEXT_CHECK_THRESHOLD = 100

        private const val BYTE_NULL: Byte = 0
        private const val BYTE_JSON_OBJECT = '{'.code.toByte()
        private const val BYTE_JSON_ARRAY = '['.code.toByte()

        private fun isJsonContent(bytes: ByteArray): Boolean {
            val firstByte = bytes.find { it > 32 }
            return firstByte == BYTE_JSON_OBJECT || firstByte == BYTE_JSON_ARRAY
        }

        private fun isText(bytes: ByteArray): Boolean =
            bytes.take(LIMIT_TEXT_CHECK_THRESHOLD).none { it == BYTE_NULL }

        private fun isMultipart(ct: String?): Boolean =
            ct?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true

        private fun isAlwaysTextField(ct: String?): Boolean {
            if (ct == null) {
                return false
            }
            val lower = ct.lowercase()
            return lower.contains(MediaType.APPLICATION_JSON_VALUE) || lower.contains(MARKER_TEXT)
        }

        private fun getFullUri(rq: HttpServletRequest): String =
            rq.queryString?.let { "${rq.requestURI}?$it" } ?: rq.requestURI

        private fun genTraceId(): String =
            UUID.randomUUID().toString().replace(STRING_DASH, STRING_EMPTY)
    }

}
