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
 * - Формирует полный URI, комбинируя baseUrl и относительный путь.
 * - Поддерживает Pretty Print для JSON-контента.
 * - Выполняет детальный разбор Multipart-запросов аналогично серверному фильтру.
 * - Эвристически определяет бинарные данные, предотвращая вывод "мусора" в логи.
 * - Исключает дублирование данных в логах при включенном DEBUG уровне.
 */
class LoggingRequestInterceptor(
    props: HttpClientProperties
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}

    /**
     * Префикс для формирования полного URI. Вычисляется один раз при создании интерцептора.
     */
    private val basePrefix: String = if (props.logFullUrl) props.url.toString().removeSuffix(SLASH) else STRING_EMPTY

    /**
     * Основной метод перехвата запроса. Управляет замером времени и выводом данных в лог.
     */
    override fun intercept(
        rq: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {

        // Ленивая подготовка данных лога запроса (вычислится только при записи в лог).
        val rqLogData by lazy { getRequestLogString(rq, body) }

        // Если DEBUG включен, печатаем данные запроса сразу.
        if (log.isDebugEnabled()) {
            log.debug { rqLogData }
        }

        val startTime = System.currentTimeMillis()
        val rs: ClientHttpResponse

        try {
            // Выполнение сетевого вызова.
            rs = execution.execute(rq, body)
        } catch (ex: Exception) {
            val duration = System.currentTimeMillis() - startTime
            // Если DEBUG выключен, выводим данные запроса в ERROR для диагностики.
            // Если DEBUG включен, запрос уже напечатан выше, выводим только ошибку и время.
            if (!log.isDebugEnabled()) {
                log.error(ex) { "External call failed! [${duration}ms]\n$rqLogData" }
            } else {
                log.error(ex) { "External call failed! [${duration}ms]" }
            }
            throw ex
        }

        val duration = System.currentTimeMillis() - startTime

        // Чтение тела ответа (требует использования BufferingClientHttpRequestFactory).
        val rsBody = rs.body.readAllBytes()
        val isError = rs.statusCode.isError

        if (isError) {
            // Если DEBUG выключен — выводим полный контекст (запрос + ответ) в ERROR.
            // Если DEBUG включен — данные запроса уже в логах, выводим только ответ.
            if (!log.isDebugEnabled()) {
                log.error {
                    "External service failure! [${duration}ms]\n$rqLogData\n${getResponseLogString(rq, rs, rsBody, duration)}"
                }
            } else {
                log.debug { "External service failure [${duration}ms]\n${getResponseLogString(rq, rs, rsBody, duration)}" }
                log.error { "External service failure detected! [${duration}ms]. See DEBUG logs for details." }
            }
        } else if (log.isDebugEnabled()) {
            // В штатном режиме при успехе выводим данные только в DEBUG.
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

                    val partHeaders = headerLines.associate {
                        it.substringBefore(STRING_COLON_SPACE) to it.substringAfter(STRING_COLON_SPACE)
                    }

                    val content = lines.dropWhile { it.isNotBlank() }.drop(1).joinToString(STRING_NEW_LINE)
                    val contentBytes = content.toByteArray()

                    var name = PART_UNKNOWN
                    var fileName: String? = null
                    var partCt: String? = null

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

                    val headersInfo = if (partHeaders.isNotEmpty()) " | Headers: $partHeaders" else STRING_EMPTY

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
     * Форматирует тело сообщения (JSON Pretty Print или текст с ограничением длины).
     */
    private fun formatBody(content: ByteArray, ct: String?): String {
        if (content.isEmpty()) {
            return BODY_EMPTY
        }

        // Защита от логирования бинарных данных через поиск NULL-байтов.
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
     * Проверяет, является ли переданный Content-Type текстовым.
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
     * Быстрая проверка первых значащих байтов на соответствие JSON ( { или [ ).
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
     * Проверяет заголовок на соответствие Multipart-типу.
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
