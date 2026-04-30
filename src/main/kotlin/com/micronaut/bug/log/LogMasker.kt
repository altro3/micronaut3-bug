package com.micronaut.bug.log

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.ObjectNode
import com.micronaut.bug.log.config.LogProperties

/**
 * Высокопроизводительный сервис маскирования данных.
 * Предназначен для очистки логов от чувствительной информации перед отправкой во внешние системы (OTLP).
 */
class LogMasker(
    props: LogProperties,
    private val mapper: ObjectMapper
) {

    private val maskingProps = props.masking

    /**
     * Кэшированные списки полей для O(1) поиска.
     */
    private val fullFields = maskingProps.full.map { it.lowercase() }.toHashSet()
    private val partialFields = maskingProps.partial.map { it.lowercase() }.toHashSet()
    private val sensitiveHeaders = maskingProps.sensitiveHeaders.map { it.lowercase() }.toHashSet()
    private val indirectHeaders = maskingProps.indirectHeaders.map { it.lowercase() }.toHashSet()

    /**
     * Кэшированные строки маскирования.
     */
    private val staticMask = maskingProps.maskChar.repeat(STATIC_MASK_LEN)
    private val partialMask = maskingProps.maskChar.repeat(PARTIAL_MASK_LEN)

    /**
     * Маскирует сообщение, если оно является отчетом ServerLoggingFilter.
     * Используется в OtlpEncoder для очистки внешних логов.
     *
     * @param message Исходное сообщение лога.
     * @return Маскированное сообщение.
     */
    fun maskServiceMessage(message: String?): String {
        if (message.isNullOrBlank() || !maskingProps.enabled) {
            return message ?: STRING_EMPTY
        }

        if (!message.contains(MARKER_SERVICE_LOG)) {
            return maskMessage(message)
        }

        var result = message

        result = maskTextSection(result, PREFIX_HEADERS, SUFFIX_HEADERS) { rawHeaders ->
            maskRawHeadersString(rawHeaders)
        }

        result = maskTextSection(result, PREFIX_BODY, SUFFIX_BODY) { rawBody ->
            maskMessage(rawBody)
        }

        return result
    }

    /**
     * Маскирует произвольное сообщение (JSON или простой текст).
     */
    fun maskMessage(message: String?): String {
        if (message.isNullOrBlank() || !maskingProps.enabled) {
            return message ?: STRING_EMPTY
        }

        val trimmed = message.trim()
        if (!(trimmed.startsWith(JSON_OBJECT_START) || trimmed.startsWith(JSON_ARRAY_START))) {
            return message
        }

        return runCatching {
            val tree = mapper.readTree(trimmed)
            mask(tree)
            mapper.writeValueAsString(tree)
        }.getOrElse {
            message
        }
    }

    /**
     * Универсальный метод маскирования карт (MDC, заголовки).
     */
    fun maskMap(data: Map<String, String>): Map<String, String> {
        if (!maskingProps.enabled || data.isEmpty()) {
            return data
        }

        return data.mapValues { (key, value) ->
            val lowerKey = key.lowercase()
            when {
                // 1. Полное затирание (секреты)
                fullFields.contains(lowerKey) || sensitiveHeaders.contains(lowerKey) -> {
                    staticMask
                }
                // 2. Частичное затирание (ПДн)
                partialFields.contains(lowerKey) -> {
                    fastPartialMask(value)
                }
                // 3. Косвенное маскирование (IP, UA, Referer)
                indirectHeaders.contains(lowerKey) -> {
                    maskIndirect(value)
                }

                else -> {
                    value
                }
            }
        }
    }

    /**
     * Рекурсивно маскирует JSON-дерево "на месте".
     */
    fun mask(node: JsonNode) {
        if (!maskingProps.enabled) {
            return
        }
        processNode(node)
    }

    private fun processNode(node: JsonNode) {
        when (node) {
            is ObjectNode -> {
                for (prop in node.properties()) {
                    val key = prop.key
                    val value = prop.value
                    val lowerKey = key.lowercase()

                    when {
                        fullFields.contains(lowerKey) -> {
                            node.put(key, staticMask)
                        }

                        partialFields.contains(lowerKey) && value.isTextual -> {
                            node.put(key, fastPartialMask(value.asText()))
                        }

                        value.isContainerNode -> {
                            processNode(value)
                        }
                    }
                }
            }

            is ArrayNode -> {
                for (i in 0 until node.size()) {
                    val item = node.get(i)
                    if (item.isContainerNode) {
                        processNode(item)
                    }
                }
            }
        }
    }

    private fun maskTextSection(text: String, start: String, end: String, masker: (String) -> String): String {
        val startIndex = text.indexOf(start)
        if (startIndex == -1) {
            return text
        }

        val contentStart = startIndex + start.length
        val endIndex = text.indexOf(end, contentStart)

        val original = if (endIndex != -1) {
            text.substring(contentStart, endIndex)
        } else {
            text.substring(contentStart)
        }

        return text.replace(original, masker(original.trim()))
    }

    private fun maskRawHeadersString(raw: String): String {
        val cleanRaw = raw.trim(CHAR_BRACE_START, CHAR_BRACE_END)
        if (cleanRaw.isBlank()) {
            return STRING_EMPTY
        }

        val map = cleanRaw.split(DELIMITER_HEADERS).associate {
            val parts = it.split(DELIMITER_KEY_VALUE, limit = 2)
            (parts.getOrNull(0) ?: STRING_EMPTY) to (parts.getOrNull(1) ?: STRING_EMPTY)
        }
        return maskMap(map).toString()
    }

    private fun maskIndirect(v: String): String {
        return if (v.length > INDIRECT_VISIBLE_LEN) {
            "${v.take(INDIRECT_VISIBLE_LEN)}$staticMask"
        } else {
            staticMask
        }
    }

    private fun fastPartialMask(v: String): String {
        val len = v.length
        if (len <= THRESHOLD_MIN_LEN) {
            return staticMask
        }
        val visible = if (len > THRESHOLD_LONG_LEN) {
            2
        } else {
            1
        }
        return "${v.take(visible)}$partialMask${v.takeLast(visible)}"
    }

    companion object {
        private const val STATIC_MASK_LEN = 6
        private const val PARTIAL_MASK_LEN = 4
        private const val THRESHOLD_MIN_LEN = 5
        private const val THRESHOLD_LONG_LEN = 8
        private const val INDIRECT_VISIBLE_LEN = 4

        private const val JSON_OBJECT_START = "{"
        private const val JSON_ARRAY_START = "["
        private const val STRING_EMPTY = ""

        private const val MARKER_SERVICE_LOG = "=================="
        private const val PREFIX_HEADERS = "Headers: "
        private const val SUFFIX_HEADERS = "\n"
        private const val PREFIX_BODY = "Body:\n"
        private const val SUFFIX_BODY = "\n=================="

        private const val DELIMITER_HEADERS = ", "
        private const val DELIMITER_KEY_VALUE = "="
        private const val CHAR_BRACE_START = '{'
        private const val CHAR_BRACE_END = '}'
    }
}
