package com.micronaut.bug.log

import com.micronaut.bug.log.config.LogProperties
import java.util.regex.Pattern

/**
 * Сверхвысокопроизводительный сервис маскирования данных.
 *
 * Предназначен для очистки логов от чувствительной информации и персональных данных (ПДн)
 * перед их отправкой во внешние системы агрегации логов (VictoriaLogs, Grafana Loki).
 *
 * Оптимизации производительности:
 * - Замена полей в JSON производится на уровне быстрых Регулярных Выражений (без тяжелого парсинга в JsonNode).
 * - Паттерн поиска JSON-полей статичен, что снимает нагрузку с Regex-движка (в отличие от динамических паттернов).
 * - Все методы маскирования пишут напрямую в общий [StringBuilder], что предотвращает конкатенацию строк.
 * - Полностью отсутствуют промежуточные аллокации строк (String.substring) в горячих циклах разбора заголовков.
 *
 * @param props Конфигурационные свойства логирования, содержащие списки полей для маскирования.
 */
class LogMasker(
    props: LogProperties,
) {

    private val maskingProps = props.masking

    /**
     * Кэшированные списки полей в нижнем регистре для обеспечения поиска за O(1).
     */
    private val fullFields = maskingProps.full.map { it.lowercase() }.toHashSet()
    private val partialFields = maskingProps.partial.map { it.lowercase() }.toHashSet()
    private val sensitiveHeaders = maskingProps.sensitiveHeaders.map { it.lowercase() }.toHashSet()
    private val indirectHeaders = maskingProps.indirectHeaders.map { it.lowercase() }.toHashSet()

    /**
     * Кэшированные статические строки масок для предотвращения повторных аллокаций.
     */
    private val staticMask = maskingProps.maskChar.repeat(STATIC_MASK_LEN)
    private val partialMask = maskingProps.maskChar.repeat(PARTIAL_MASK_LEN)

    /**
     * Статичное регулярное выражение для поиска JSON-полей.
     * Позволяет находить любые пары "ключ":"значение" без перебора конкретных ключей на уровне Regex.
     *
     * Группы захвата:
     * - Группа 1: Имя ключа (без кавычек).
     * - Группа 2: Тип кавычки значения (одинарная или двойная).
     * - Группа 3: Текстовое содержимое значения.
     */
    private val jsonPattern: Pattern = Pattern.compile(
        "\"([^\"]+)\"\\s*:\\s*([\"'])(.*?)\\2",
        Pattern.CASE_INSENSITIVE
    )

    /**
     * Маскирует сообщение, если оно является структурированным отчетом от [ServerLoggingFilter].
     * Внутри отчета маскируются секции заголовков и тела ответа по разным правилам.
     *
     * @param message Исходное сообщение лога.
     * @return Полностью маскированное сообщение или исходное, если маскирование выключено.
     */
    fun maskServiceMessage(message: String?): String {
        if (message.isNullOrBlank() || !maskingProps.enabled) {
            return message ?: STRING_EMPTY
        }

        // Если это обычный лог, а не структурированный отчёт фильтра — маскируем как простой текст
        if (message.indexOf(MARKER_SERVICE_LOG) == -1) {
            return maskMessage(message)
        }

        val sb = StringBuilder(message.length)

        val headersStart = message.indexOf(PREFIX_HEADERS)
        val bodyStart = message.indexOf(PREFIX_BODY)

        if (headersStart == -1 && bodyStart == -1) {
            return maskMessage(message)
        }

        // 1. Копируем блок данных до начала первой секции логирования
        val firstSectionEnd = if (headersStart != -1) headersStart else bodyStart
        sb.append(message, 0, firstSectionEnd)

        // 2. Обрабатываем секцию Headers
        if (headersStart != -1) {
            sb.append(PREFIX_HEADERS)
            val contentStart = headersStart + PREFIX_HEADERS.length
            val contentEnd = message.indexOf(SUFFIX_HEADERS, contentStart)

            val endOfSection = if (contentEnd != -1) contentEnd else message.length

            maskRawHeadersFast(message, contentStart, endOfSection, sb)

            if (contentEnd != -1) {
                val nextStart = if (bodyStart != -1 && bodyStart > contentEnd) bodyStart else contentEnd
                sb.append(message, contentEnd, nextStart)
            }
        }

        // 3. Обрабатываем секцию Body
        if (bodyStart != -1) {
            sb.append(PREFIX_BODY)
            val contentStart = bodyStart + PREFIX_BODY.length
            val contentEnd = message.indexOf(SUFFIX_BODY, contentStart)

            val endOfSection = if (contentEnd != -1) contentEnd else message.length

            maskMessageSequence(message, contentStart, endOfSection, sb)

            if (contentEnd != -1) {
                sb.append(message, contentEnd, message.length)
            }
        }

        return sb.toString()
    }

    /**
     * Маскирует произвольное текстовое сообщение (полный аналог [maskMessageSequence],
     * но работающий со всей строкой целиком).
     *
     * @param message Исходное сообщение.
     * @return Строка с замаскированными полями.
     */
    fun maskMessage(message: String): String {
        val sb = StringBuilder(message.length)
        maskMessageSequence(message, 0, message.length, sb)
        return sb.toString()
    }

    /**
     * Маскирует JSON-подобный текст в заданной области строки напрямую в [StringBuilder].
     *
     * Метод использует [Matcher.region] для ограничения области поиска регулярным выражением,
     * предотвращая копирование подстрок через substring.
     *
     * @param text Исходный текст для анализа.
     * @param start Индекс начала области обработки.
     * @param end Индекс конца области обработки.
     * @param sb Результирующий билдер для сборки маскированного текста.
     */
    private fun maskMessageSequence(text: String, start: Int, end: Int, sb: StringBuilder) {
        val matcher = jsonPattern.matcher(text)
        matcher.region(start, end)
        matcher.reset()

        var lastAppendPosition = start
        val keyBuilder = StringBuilder(32)

        while (matcher.find()) {
            val keyStart = matcher.start(1)
            val keyEnd = matcher.end(1)

            // Безаллокационное приведение ключа к нижнему регистру для сверки со словарями
            keyBuilder.setLength(0)
            for (i in keyStart until keyEnd) {
                keyBuilder.append(text[i].lowercaseChar())
            }
            val key = keyBuilder.toString()

            val isFull = fullFields.contains(key)
            val isPartial = partialFields.contains(key)

            if (isFull || isPartial) {
                // Копируем неизмененный текст до текущего найденного ключа
                sb.append(text, lastAppendPosition, matcher.start())

                val quote = matcher.group(2)
                val valStart = matcher.start(3)
                val valEnd = matcher.end(3)

                sb.append(CHAR_QUOTE).append(key).append(DELIMITER_JSON_KEY_VALUE).append(quote)

                if (isFull) {
                    sb.append(staticMask)
                } else {
                    appendFastPartialMask(text, valStart, valEnd, sb)
                }

                sb.append(quote)
                lastAppendPosition = matcher.end()
            }
        }
        // Дописываем остаток строки
        sb.append(text, lastAppendPosition, end)
    }

    /**
     * Потоковый линейный разбор строки HTTP-заголовков на индексах.
     * Парсит строку вида `{Header1=Val1, Header2=Val2}` без выделения промежуточных объектов в Heap.
     *
     * @param text Исходный текст, содержащий строку заголовков.
     * @param start Индекс начала секции заголовков.
     * @param end Индекс конца секции заголовков.
     * @param sb Результирующий билдер.
     */
    private fun maskRawHeadersFast(text: String, start: Int, end: Int, sb: StringBuilder) {
        var current = start
        var limit = end
        while (current < limit && text[current] <= CHAR_SPACE) {
            current++
        }
        while (limit > current && text[limit - 1] <= CHAR_SPACE) {
            limit--
        }

        if (current < limit && text[current] == CHAR_BRACE_START) {
            current++
        }
        if (limit > current && text[limit - 1] == CHAR_BRACE_END) {
            limit--
        }

        sb.append(CHAR_BRACE_START)

        var isFirst = true
        val keyBuilder = StringBuilder(32)

        while (current < limit) {
            val delimiterIndex = text.indexOf(DELIMITER_HEADERS, current)
            val pairEnd = if (delimiterIndex != -1 && delimiterIndex < limit) delimiterIndex else limit

            val equalsIndex = text.indexOf(DELIMITER_KEY_VALUE, current)

            if (equalsIndex != -1 && equalsIndex < pairEnd) {
                var keyStart = current
                while (keyStart < equalsIndex && text[keyStart] <= CHAR_SPACE) {
                    keyStart++
                }
                var keyEnd = equalsIndex
                while (keyEnd > keyStart && text[keyEnd - 1] <= CHAR_SPACE) {
                    keyEnd--
                }

                keyBuilder.setLength(0)
                for (i in keyStart until keyEnd) {
                    keyBuilder.append(text[i].lowercaseChar())
                }
                val key = keyBuilder.toString()

                val valStart = equalsIndex + 1
                val valEnd = pairEnd

                if (!isFirst) {
                    sb.append(DELIMITER_HEADERS)
                }
                sb.append(text, keyStart, keyEnd).append(DELIMITER_KEY_VALUE)

                when {
                    fullFields.contains(key) || sensitiveHeaders.contains(key) -> {
                        sb.append(staticMask)
                    }
                    partialFields.contains(key) -> {
                        appendFastPartialMask(text, valStart, valEnd, sb)
                    }
                    indirectHeaders.contains(key) -> {
                        appendMaskIndirect(text, valStart, valEnd, sb)
                    }
                    else -> {
                        sb.append(text, valStart, valEnd)
                    }
                }
                isFirst = false
            }

            if (delimiterIndex == -1 || delimiterIndex >= limit) {
                break
            }
            current = delimiterIndex + DELIMITER_HEADERS.length
        }

        sb.append(CHAR_BRACE_END)
    }

    /**
     * Маскирует данные в классической карте [Map].
     * Применяется для очистки MDC-контекста перед отправкой в OTLP.
     *
     * @param data Исходная мапа (например, MDC).
     * @return Новая мапа с замаскированными значениями.
     */
    fun maskMap(data: Map<String, String>): Map<String, String> {
        if (!maskingProps.enabled || data.isEmpty()) {
            return data
        }
        return data.mapValues { (key, value) ->
            val lowerKey = key.lowercase()
            when {
                fullFields.contains(lowerKey) || sensitiveHeaders.contains(lowerKey) -> {
                    staticMask
                }
                partialFields.contains(lowerKey) -> {
                    fastPartialMask(value)
                }
                indirectHeaders.contains(lowerKey) -> {
                    if (value.length > INDIRECT_VISIBLE_LEN) {
                        "${value.substring(0, INDIRECT_VISIBLE_LEN)}$staticMask"
                    } else {
                        staticMask
                    }
                }
                else -> {
                    value
                }
            }
        }
    }

    /**
     * Безаллокационное косвенное маскирование. Пишет результат сразу в билдер.
     * Оставляет видимыми только первые [INDIRECT_VISIBLE_LEN] символов.
     */
    private fun appendMaskIndirect(src: String, start: Int, end: Int, sb: StringBuilder) {
        val len = end - start
        if (len > INDIRECT_VISIBLE_LEN) {
            sb.append(src, start, start + INDIRECT_VISIBLE_LEN).append(staticMask)
        } else {
            sb.append(staticMask)
        }
    }

    /**
     * Безаллокационное частичное маскирование. Пишет результат сразу в билдер.
     * Прячет середину строки, оставляя видимыми начало и конец.
     */
    private fun appendFastPartialMask(src: String, start: Int, end: Int, sb: StringBuilder) {
        val len = end - start
        if (len <= THRESHOLD_MIN_LEN) {
            sb.append(staticMask)
            return
        }
        val visible = if (len > THRESHOLD_LONG_LEN) 2 else 1
        sb.append(src, start, start + visible)
            .append(partialMask)
            .append(src, end - visible, end)
    }

    /**
     * Классическое частичное маскирование строки с выделением памяти.
     * Используется только в методе [maskMap], где выделение новой строки неизбежно.
     */
    private fun fastPartialMask(v: String): String {
        val len = v.length
        if (len <= THRESHOLD_MIN_LEN) {
            return staticMask
        }

        val visible = if (len > THRESHOLD_LONG_LEN) 2 else 1

        val sb = StringBuilder(len)
        sb.append(v, 0, visible)
            .append(partialMask)
            .append(v, len - visible, len)

        return sb.toString()
    }

    companion object {
        private const val STATIC_MASK_LEN = 6
        private const val PARTIAL_MASK_LEN = 4
        private const val THRESHOLD_MIN_LEN = 5
        private const val THRESHOLD_LONG_LEN = 8
        private const val INDIRECT_VISIBLE_LEN = 4
        private const val STRING_EMPTY = ""
        private const val MARKER_SERVICE_LOG = "=================="
        private const val PREFIX_HEADERS = "Headers: "
        private const val SUFFIX_HEADERS = "\n"
        private const val PREFIX_BODY = "Body:\n"
        private const val SUFFIX_BODY = "\n=================="
        private const val DELIMITER_HEADERS = ", "
        private const val DELIMITER_KEY_VALUE = "="
        private const val DELIMITER_JSON_KEY_VALUE = "\":"
        private const val CHAR_BRACE_START = '{'
        private const val CHAR_BRACE_END = '}'
        private const val CHAR_QUOTE = '"'
        private const val CHAR_SPACE = ' '
    }
}
