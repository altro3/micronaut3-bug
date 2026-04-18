package com.micronaut.bug.client

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.HttpRequest
import org.springframework.http.MediaType
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse

/**
 * Интерцептор для детального логирования исходящих HTTP-запросов и ответов.
 *
 * Основные возможности:
 * - Поддержка Pretty Print для JSON-контента.
 * - Детальный разбор Multipart-запросов с отображением метаданных частей.
 * - Эвристическая проверка на бинарные данные для предотвращения повреждения логов.
 * - Измерение времени выполнения запроса (Duration).
 * - Группировка запроса и ответа в одном событии при возникновении ошибок.
 */
class LoggingRequestInterceptor(
    private val props: HttpClientProperties
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}

    /**
     * Префикс для формирования полного URI.
     * Если логирование полного пути отключено, префикс остается пустым.
     */
    private val basePrefix: String = if (props.logFullUrl) props.url.toString().removeSuffix(SLASH) else STRING_EMPTY

    /**
     * Перехватывает выполнение HTTP-запроса.
     * Замеряет время выполнения и выводит логи в зависимости от статуса ответа.
     */
    override fun intercept(
        rq: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {

        // Используем lazy для подготовки строки лога запроса.
        // Если логирование уровня DEBUG выключено, ресурсоемкие операции по сборке строки не выполнятся.
        val rqLogData by lazy { getRequestLogString(rq, body) }

        if (log.isDebugEnabled()) {
            log.debug { rqLogData }
        }

        val startTime = System.currentTimeMillis()
        val rs: ClientHttpResponse

        try {
            // Выполнение запроса далее по цепочке интерцепторов
            rs = execution.execute(rq, body)
        } catch (ex: Exception) {
            val duration = System.currentTimeMillis() - startTime
            // При сетевых ошибках (Timeout, Connection Refused) выводим контекст запроса на уровне ERROR
            log.error(ex) { "External call failed! [${duration}ms]\n$rqLogData" }
            throw ex
        }

        val duration = System.currentTimeMillis() - startTime

        // Чтение тела ответа. Требует наличия BufferingClientHttpRequestFactory в конфигурации клиента,
        // иначе поток ответа будет закрыт после этого прочтения.
        val rsBody = rs.body.readAllBytes()
        val isError = rs.statusCode.isError

        if (isError) {
            // В случае ошибки (4xx, 5xx) логируем и запрос, и ответ вместе для упрощения отладки в ELK/Splunk
            log.error {
                "External service error! [${duration}ms]\n$rqLogData\n${getResponseLogString(rq, rs, rsBody, duration)}"
            }
        } else if (log.isDebugEnabled()) {
            // В штатном режиме выводим только данные ответа в DEBUG
            log.debug { "External service success [${duration}ms]\n${getResponseLogString(rq, rs, rsBody, duration)}" }
        }

        return rs
    }

    /**
     * Формирует текстовый блок данных исходящего запроса.
     */
    private fun getRequestLogString(rq: HttpRequest, body: ByteArray): String {
        val ct = rq.headers.contentType?.toString()
        val bodyResult = if (isMultipart(ct)) {
            formatMultipart(body, ct)
        } else {
            formatBody(body, ct)
        }

        return """
            |================== Client request ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |Headers: ${rq.headers}
            |Body:
            |$bodyResult
            |================== /Client request ==================
        """.trimMargin()
    }

    /**
     * Формирует текстовый блок данных входящего ответа.
     */
    private fun getResponseLogString(rq: HttpRequest, rs: ClientHttpResponse, body: ByteArray, duration: Long): String {
        val ct = rs.headers.contentType?.toString()
        val bodyResult = if (isMultipart(ct)) {
            formatMultipart(body, ct)
        } else {
            formatBody(body, ct)
        }

        return """
            |================== Client response ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |Status: ${rs.statusCode} (${rs.statusText})
            |Duration: ${duration}ms
            |Headers: ${rs.headers}
            |Body:
            |$bodyResult
            |================== /Client response ==================
        """.trimMargin()
    }

    /**
     * Выполняет разбор массива байтов Multipart-сообщения.
     * Разделяет тело по границе (boundary), извлекает заголовки и контент каждой части.
     */
    private fun formatMultipart(bytes: ByteArray, ct: String?): String {
        if (bytes.isEmpty()) {
            return BODY_EMPTY
        }

        // Пытаемся извлечь маркер границы из заголовка Content-Type
        val boundary = ct?.split(MARKER_BOUNDARY)?.getOrNull(1)?.let {
            PREFIX_DASH + it
        } ?: return BODY_MULTIPART_RAW

        return runCatching {
            bytes.toString(Charsets.UTF_8).split(boundary)
                .filter { it.contains(MARKER_CONTENT_DISPOSITION) }
                .joinToString(STRING_NEW_LINE) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headerLines = lines.takeWhile { it.isNotBlank() }

                    // Сбор локальных заголовков конкретной части Multipart
                    val partHeaders = headerLines.associate {
                        it.substringBefore(STRING_COLON_SPACE) to it.substringAfter(STRING_COLON_SPACE)
                    }

                    // Содержимое части (тело) находится после пустой строки-разделителя
                    val content = lines.dropWhile { it.isNotBlank() }.drop(1).joinToString(STRING_NEW_LINE)
                    val contentBytes = content.toByteArray()

                    var name = PART_UNKNOWN
                    var fileName: String? = null
                    var partCt: String? = null

                    // Извлечение метаданных из заголовков части
                    partHeaders.forEach { (k, v) ->
                        if (k.contains(MARKER_CONTENT_DISPOSITION)) {
                            name = v.substringAfter(EQUALS_NAME, "").substringBefore(STRING_QUOTE)
                            if (v.contains(MARKER_FILENAME)) {
                                fileName = v.substringAfter(EQUALS_FILENAME).substringBefore(STRING_QUOTE)
                            }
                        } else if (k.equals(HEADER_CONTENT_TYPE, ignoreCase = true)) {
                            partCt = v
                        }
                    }

                    val description = if (fileName != null) {
                        TEMPLATE_PART_FILE_INFO.format(name, fileName, contentBytes.size.toLong())
                    } else {
                        name
                    }

                    val headersInfo = if (partHeaders.isNotEmpty()) {
                        " | Headers: $partHeaders"
                    } else {
                        STRING_EMPTY
                    }

                    // Проверка: если часть является файлом и содержит бинарный контент — логируем только метаданные
                    if (fileName != null && !isAlwaysTextField(partCt) && !isText(contentBytes)) {
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, BODY_BINARY)
                    } else {
                        val formattedContent = formatBody(contentBytes, partCt)
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, "Content: $formattedContent")
                    }
                }
        }.getOrElse { BODY_MULTIPART_RAW }
    }

    /**
     * Форматирует тело сообщения.
     * Если определен JSON — делает Pretty Print, иначе возвращает строку (с ограничением по длине).
     */
    private fun formatBody(content: ByteArray, ct: String?): String {
        if (content.isEmpty()) {
            return BODY_EMPTY
        }

        // Предотвращаем вывод бинарных данных (картинки, архивы) в текстовый лог
        if (!isText(content)) {
            return BODY_BINARY
        }

        val isJson = ct?.contains(MediaType.APPLICATION_JSON_VALUE) == true || isJsonContent(content)
        if (isJson) {
            return runCatching {
                val tree = prettyMapper.readTree(content)
                prettyMapper.writeValueAsString(tree)
            }.getOrElse { content.toString(Charsets.UTF_8) }
        }

        val text = content.toString(Charsets.UTF_8)
        return if (text.length > LIMIT_LOG_SIZE) {
            text.take(LIMIT_LOG_SIZE) + SUFFIX_TRUNCATED
        } else {
            text
        }
    }

    /**
     * Собирает полный URI запроса, объединяя базовый префикс и относительный путь.
     */
    private fun getFullUri(rq: HttpRequest): String {
        val path = rq.uri.toString()
        return if (basePrefix.isNotEmpty() && !path.startsWith(MARKER_HTTP_PROTOCOL)) {
            basePrefix + path
        } else {
            path
        }
    }

    /**
     * Проверяет, является ли переданный Content-Type гарантированно текстовым (JSON или явный Text).
     */
    private fun isAlwaysTextField(ct: String?): Boolean {
        if (ct == null) {
            return false
        }
        val lower = ct.lowercase()
        return lower.contains(MediaType.APPLICATION_JSON_VALUE) || lower.contains(MARKER_TEXT)
    }

    /**
     * Проверяет начало массива байтов на наличие NULL-символов.
     * Используется для быстрого детектирования бинарных файлов.
     */
    private fun isText(bytes: ByteArray): Boolean {
        if (bytes.isEmpty()) {
            return true
        }
        val limit = minOf(bytes.size, LIMIT_TEXT_CHECK_THRESHOLD)
        for (i in 0 until limit) {
            if (bytes[i] == BYTE_ZERO) {
                return false
            }
        }
        return true
    }

    /**
     * Выполняет быструю проверку первого значащего байта на соответствие структуре JSON ( { или [ ).
     */
    private fun isJsonContent(bytes: ByteArray): Boolean {
        for (b in bytes) {
            if (b <= BYTE_SPACE) {
                continue
            }
            return b == BYTE_JSON_OBJECT || b == BYTE_JSON_ARRAY
        }
        return false
    }

    /**
     * Проверяет, является ли заголовок Content-Type признаком Multipart-запроса.
     */
    private fun isMultipart(ct: String?): Boolean {
        return ct?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE) == true
    }

    companion object {
        private val prettyMapper = ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)

        private const val STRING_EMPTY = ""
        private const val STRING_NEW_LINE = "\n"
        private const val STRING_COLON_SPACE = ": "
        private const val STRING_QUOTE = "\""
        private const val SLASH = "/"

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_MULTIPART_RAW = "[MULTIPART RAW]"

        private const val PART_UNKNOWN = "UNKNOWN"
        private const val TEMPLATE_PART_PREFIX = "  [PART] -> Name: %s%s | %s"
        private const val TEMPLATE_PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"

        private const val MARKER_BOUNDARY = "boundary="
        private const val MARKER_CONTENT_DISPOSITION = "Content-Disposition"
        private const val MARKER_FILENAME = "filename="
        private const val MARKER_TEXT = "text"
        private const val MARKER_HTTP_PROTOCOL = "http"

        private const val EQUALS_NAME = "name=\""
        private const val EQUALS_FILENAME = "filename=\""
        private const val PREFIX_DASH = "--"
        private const val HEADER_CONTENT_TYPE = "Content-Type"

        private const val SUFFIX_TRUNCATED = "... [TRUNCATED]"
        private const val LIMIT_LOG_SIZE = 8192
        private const val LIMIT_TEXT_CHECK_THRESHOLD = 100

        /** Байтовые константы для оптимизации производительности в циклах */
        private const val BYTE_ZERO = 0.toByte()
        private const val BYTE_SPACE = 32.toByte()
        private const val BYTE_JSON_OBJECT = '{'.code.toByte()
        private const val BYTE_JSON_ARRAY = '['.code.toByte()
    }
}
