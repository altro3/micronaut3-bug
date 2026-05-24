package com.altro.common.log

import com.altro.common.log.ContentTypeAnalyzer.isBinaryContent
import com.altro.common.log.ContentTypeAnalyzer.isJsonContent
import com.altro.common.log.ContentTypeAnalyzer.isMultipart
import com.altro.common.log.LogConst.BODY_BINARY
import com.altro.common.log.LogConst.BODY_EMPTY
import com.altro.common.log.LogConst.BODY_MULTIPART_RS
import com.altro.common.log.LogConst.BODY_TOO_LARGE
import com.altro.common.log.LogConst.PART_UNKNOWN
import com.altro.common.log.LogConst.STRING_EMPTY
import com.altro.common.log.LogConst.TEMPLATE_PART_FILE_INFO
import com.altro.common.log.LogConst.TEMPLATE_PART_PREFIX
import com.altro.common.log.LogUtil.truncateIfNeeded
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import tools.jackson.databind.SerializationFeature
import tools.jackson.databind.json.JsonMapper

class LogFormatter(
    jsonMapper: JsonMapper,
    private val logMasker: LogMasker? = null,
) {

    private val prettyMapper: JsonMapper = jsonMapper.rebuild()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .build()

    fun formatBody(
        content: ByteArray,
        contentType: String?,
        headers: Map<String, String>? = null,
        prettyPrint: Boolean,
        limitLogSize: Int,
        truncateChunkSize: Int,
    ): String {
        if (content.isEmpty() || content.contentEquals(BODY_EMPTY.toByteArray())) return BODY_EMPTY
        if (content.contentEquals(BODY_TOO_LARGE.toByteArray())) return BODY_TOO_LARGE

        if (prettyPrint && !isMultipart(contentType) && isBinaryContent(content, headers, contentType)) {
            return BODY_BINARY
        }

        val resultText = if (prettyPrint) formatIfJson(content, contentType) else String(content, Charsets.UTF_8)
        return truncateIfNeeded(resultText, limitLogSize, truncateChunkSize)
    }

    private fun formatIfJson(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) return STRING_EMPTY
        val isJson = contentType?.contains(MediaType.APPLICATION_JSON_VALUE) == true || isJsonContent(bytes)

        return if (isJson) {
            runCatching { prettyMapper.writeValueAsString(prettyMapper.readTree(bytes)) }.getOrElse { String(bytes) }
        } else {
            String(bytes)
        }
    }

    fun formatMultipartResponse(
        bytes: ByteArray,
        contentType: String?,
        prettyPrint: Boolean,
        limitLogSize: Int,
        truncateChunkSize: Int,
    ): String {
        if (bytes.isEmpty()) return BODY_EMPTY
        val boundary = contentType?.substringAfter(MARKER_BOUNDARY, STRING_EMPTY)
            ?.substringBefore(SEMICOLON)?.replace(QUOTE, STRING_EMPTY)?.trim() ?: return BODY_MULTIPART_RS

        val delimiter = "$NEW_LINE$DOUBLE_DASH${boundary.trim()}"

        return runCatching {
            bytes.toString(Charsets.UTF_8).split(delimiter)
                .filter { it.isNotBlank() && it != DOUBLE_DASH && it.contains(HttpHeaders.CONTENT_DISPOSITION) }
                .joinToString(NEW_LINE) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headerLines = lines.takeWhile { it.isNotBlank() }
                    val partHeaders = headerLines.associate { it.substringBefore(COLON_SPACE) to it.substringAfter(COLON_SPACE) }
                    val content = lines.dropWhile { it.isNotBlank() }.drop(1).joinToString(NEW_LINE)

                    var name: String? = null
                    var fileName: String? = null
                    var partCt: String? = null

                    partHeaders.forEach { (k, v) ->
                        when {
                            k.contains(HttpHeaders.CONTENT_DISPOSITION) -> {
                                name = v.substringAfter(EQUALS_NAME, STRING_EMPTY).substringBefore(QUOTE)
                                fileName = if (v.contains(MARKER_FILENAME)) v.substringAfter(EQUALS_FILENAME).substringBefore(QUOTE) else null
                            }
                            k.equals(HttpHeaders.CONTENT_TYPE, ignoreCase = true) -> partCt = v
                        }
                    }

                    val finalName = name ?: PART_UNKNOWN
                    val contentBytes = content.toByteArray()
                    val description = if (fileName != null) TEMPLATE_PART_FILE_INFO.format(finalName, fileName, contentBytes.size.toLong()) else finalName
                    val headersInfo = if (partHeaders.isNotEmpty()) " | Headers: $partHeaders" else STRING_EMPTY

                    if (isBinaryContent(contentBytes, partHeaders, partCt, finalName)) {
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, BODY_BINARY)
                    } else {
                        val headersMap = logMasker?.maskMap(partHeaders) ?: partHeaders
                        val partContent = formatBody(contentBytes, partCt, headersMap, prettyPrint, limitLogSize, truncateChunkSize)
                        val prefix = if (partContent == BODY_BINARY) BODY_BINARY else "Content: $partContent"
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, prefix)
                    }
                }
        }.getOrElse { BODY_MULTIPART_RS }
    }

    companion object {
        private const val QUOTE = "\""
        private const val NEW_LINE = "\n"
        private const val COLON_SPACE = ": "
        private const val SEMICOLON = ";"
        private const val DOUBLE_DASH = "--"
        private const val MARKER_BOUNDARY = "boundary="
        private const val MARKER_FILENAME = "filename="
        private const val EQUALS_NAME = "name=\""
        private const val EQUALS_FILENAME = "filename=\""
    }
}
