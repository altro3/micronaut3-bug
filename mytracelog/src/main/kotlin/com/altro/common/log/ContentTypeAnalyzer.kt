package com.altro.common.log

import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType

object ContentTypeAnalyzer {

    private const val STRING_EMPTY = ""
    private const val MARKER_TEXT = "text"
    private const val EQUALS_NAME = "name=\""
    private const val LIMIT_TEXT_CHECK_THRESHOLD = 512
    private const val BYTE_JSON_OBJECT = '{'.code.toByte()
    private const val BYTE_JSON_ARRAY = '['.code.toByte()

    private val COMPRESSION_ENCODINGS = setOf("gzip", "br", "deflate")
    private val EXTENSIONS_TEXT = setOf(
        "txt", "json", "xml", "html", "csv", "yml", "yaml",
        "log", "toml", "properties", "conf", "config",
        "md", "sql", "js", "css", "sh", "bat", "py"
    )
    private val BINARY_MIME_MARKERS = setOf(
        "octet-stream", "image/", "video/", "audio/", "pdf", "zip",
        "vnd.ms-excel", "vnd.openxmlformats-officedocument", "application/msword"
    )

    fun isJsonContent(bytes: ByteArray): Boolean {
        val size = bytes.size
        for (i in 0 until size) {
            val b = bytes[i]
            if (b <= 32) continue
            return b == BYTE_JSON_OBJECT || b == BYTE_JSON_ARRAY
        }
        return false
    }

    fun isBinaryContent(
        bytes: ByteArray,
        headers: Map<String, String>? = null,
        contentType: String? = null,
        fileName: String? = null
    ): Boolean {
        val encoding = headers?.get(HttpHeaders.CONTENT_ENCODING)
        if (!encoding.isNullOrEmpty() && COMPRESSION_ENCODINGS.any { encoding.contains(it, ignoreCase = true) } || isGzipSignature(bytes)) {
            return true
        }

        if (isAlwaysTextField(contentType)) {
            return false
        }

        if (contentType?.lowercase()?.let { ct -> BINARY_MIME_MARKERS.any { ct.contains(it) } } == true) {
            return true
        }

        val effectiveFileName = fileName ?: contentType?.substringAfter(EQUALS_NAME, STRING_EMPTY)?.substringBefore("\"", "")?.takeIf { it.isNotBlank() }
        if (effectiveFileName != null) {
            val ext = effectiveFileName.substringAfterLast('.', STRING_EMPTY).lowercase()
            if (ext.isNotEmpty() && ext !in EXTENSIONS_TEXT) {
                return true
            }
        }

        return !isText(bytes)
    }

    private fun isGzipSignature(b: ByteArray): Boolean =
        b.size >= 2 && (b[0].toInt() and 0xFF == 0x1F) && (b[1].toInt() and 0xFF == 0x8B)

    private fun isText(bytes: ByteArray): Boolean {
        if (bytes.isEmpty()) return true
        val limit = minOf(bytes.size, LIMIT_TEXT_CHECK_THRESHOLD)

        for (i in 0 until limit) {
            val b = bytes[i].toInt() and 0xFF
            if (b == 0) return false
            if (b < 32 && b != 9 && b != 10 && b != 13) {
                return false
            }
        }
        return true
    }

    fun isMultipart(ct: String?): Boolean =
        ct?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true

    private fun isAlwaysTextField(ct: String?): Boolean {
        if (ct == null) return false
        val lower = ct.lowercase()
        return lower.contains(MediaType.APPLICATION_JSON_VALUE) || lower.contains(MARKER_TEXT)
    }
}
