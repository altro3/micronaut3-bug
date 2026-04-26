package com.micronaut.bug.client

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.LIMIT_TEXT_CHECK_THRESHOLD
import io.github.oshai.kotlinlogging.KotlinLogging
import org.jboss.logging.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.CONTENT_DISPOSITION
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpRequest
import org.springframework.http.MediaType
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse
import java.io.ByteArrayInputStream
import java.util.zip.GZIPInputStream

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
    props: HttpClientProperties,
    objectMapper: ObjectMapper,
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}

    private val logProps = props.log
    private val prettyMapper = objectMapper.copy()
        .enable(SerializationFeature.INDENT_OUTPUT)

    /**
     * Префикс для формирования полного URI. Вычисляется один раз при создании интерцептора.
     */
    private val basePrefix: String = if (logProps.fullUrl) props.url.toString().removeSuffix(SLASH) else STRING_EMPTY

    /**
     * Основной метод перехвата запроса. Управляет замером времени и выводом данных в лог.
     */
    override fun intercept(
        rq: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {
        val isDebug = log.isDebugEnabled()

        val skipLogging = rq.attributes[ATTR_SKIP_LOGGING] as? Boolean ?: false

        try {
            // Достаём ID из атрибутов, чтобы GzipRequestInterceptor его увидел
            val extRqId = rq.attributes.getValue(ATTR_EXT_RQ_ID).toString()

            // Данные запроса готовим лениво
            val rqLogData by lazy { getRequestLogString(rq, body, extRqId, skipLogging) }

            // 1. Включен DEBUG: логируем запрос сразу перед отправкой
            if (isDebug) {
                log.debug { rqLogData }
            }

            val startTime = System.currentTimeMillis()
            val rs: ClientHttpResponse
            try {
                rs = execution.execute(rq, body)
            } catch (e: Exception) {
                val duration = System.currentTimeMillis() - startTime
                if (isDebug) {
                    // В DEBUG запрос уже есть в логах, пишем только ID и ошибку
                    log.error(e) { "External call failed! [extRqId: $extRqId, duration: ${duration}ms]" }
                } else {
                    // Без DEBUG выводим всё вместе: и данные запроса, и ошибку
                    log.error(e) { "External call failed! [extRqId: $extRqId, duration: ${duration}ms]\n$rqLogData" }
                }
                throw e
            }

            val duration = System.currentTimeMillis() - startTime

            val isError = rs.statusCode.isError

            if (isDebug || isError) {
                // 1. Просто получаем сырые байты (или заглушку)
                val rsBodyBytes = if (skipLogging) {
                    BODY_LOG_DISABLED.toByteArray()
                } else {

                    val contentLength = rs.headers.contentLength
                    val maxAllowed = logProps.maxPayloadSize.toBytes()

                    // ПРЕДОХРАНИТЕЛЬ: Проверяем заголовок ДО чтения
                    if (contentLength > maxAllowed) {
                        BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)
                    } else {
                        // Если заголовок -1 (chunked) или в рамках лимита — читаем,
                        // но ограничиваем само чтение, чтобы не доверять заголовку на 100%
                        val rawBytes = rs.body.readAllBytes()

                        if (rawBytes.size > maxAllowed) {
                            BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)
                        } else {
                            rawBytes
                        }
                    }
                }

                // 2. Основной лог ответа
                if (isDebug) {
                    // В DEBUG всегда пишем ответ отдельно (статус ошибки будет внутри шаблона)
                    log.debug { getResponseLogString(rq, rs, rsBodyBytes, duration, extRqId, skipLogging) }
                } else {
                    // Без DEBUG логируем только ошибки: запрос и ответ одним блоком
                    log.error { "External service failure!\n$rqLogData\n${getResponseLogString(rq, rs, rsBodyBytes, duration, extRqId, skipLogging)}" }
                }
            }
            return rs
        } finally {
            MDC.remove(MDC_NEW_EXT_RQ_ID)
        }
    }

    /**
     * Проверяет наличие заголовка Content-Encoding: gzip и распаковывает байты.
     *
     * @param response ответ сервера для проверки заголовков
     * @param bytes сырые байты тела
     * @return распакованный массив байт или оригинальный, если сжатие отсутствует
     */
    private fun decompressIfNeeded(response: ClientHttpResponse, bytes: ByteArray): ByteArray {
        val isGzip = response.headers.getFirst(HttpHeaders.CONTENT_ENCODING)?.contains(ENCODING_GZIP, true) == true
        return if (isGzip && bytes.isNotEmpty()) {
            try {
                GZIPInputStream(ByteArrayInputStream(bytes)).use { it.readBytes() }
            } catch (e: Exception) {
                log.warn { "Failed to decompress GZIP body, logging raw data. Error: ${e.message}" }
                bytes
            }
        } else {
            bytes
        }
    }

    /**
     * Формирует текстовый блок данных исходящего запроса.
     */
    private fun getRequestLogString(rq: HttpRequest, body: ByteArray, extRqId: String, skipLogging: Boolean): String {
        val ct = rq.headers.contentType?.toString()
        val bodyForLog = if (body.size > logProps.maxPayloadSize.toBytes() && !skipLogging) {
            BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)
        } else {
            body
        }
        val bodyResult = if (skipLogging) {
            BODY_LOG_DISABLED
        } else if (logProps.prettyPrint && isMultipart(ct)) {
            formatMultipart(bodyForLog, ct)
        } else {
            formatBody(bodyForLog, ct, rq.headers.toSingleValueMap())
        }

        return """
            |
            |================== Client request ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |ExtRqId: $extRqId
            |Headers: ${rq.headers}
            |Body:
            |$bodyResult
            |================== /Client request ==================
        """.trimMargin()
    }

    /**
     * Формирует текстовый блок данных входящего ответа.
     */
    private fun getResponseLogString(rq: HttpRequest, rs: ClientHttpResponse, body: ByteArray, duration: Long, extRqId: String, skipLogging: Boolean): String {
        val ct = rs.headers.contentType?.toString()
        // 1. Сначала решаем: нужно ли нам вообще трогать байты (распаковывать)
        val processedBody = if (!logProps.prettyPrint || skipLogging) {
            body // В сыром режиме или при skipLogging оставляем как есть
        } else {
            // Только если включен prettyPrint — пытаемся разжать для анализа
            decompressIfNeeded(rs, body)
        }

        // 2. Формируем строку для лога
        val bodyResult = if (skipLogging) {
            BODY_LOG_DISABLED
        } else if (isMultipart(ct)) {
            formatMultipart(processedBody, ct)
        } else {
            formatBody(processedBody, ct, rs.headers.toSingleValueMap()) // formatBody сам проверит isText
        }

        return """
            |
            |================== Client response ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |ExtRqId: $extRqId
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
     *
     * Если [prettyPrint] включен, разбивает тело на части по границе (boundary),
     * извлекает заголовки каждой части и форматирует контент.
     * Для каждой части выполняется индивидуальная проверка на бинарные данные.
     *
     * @param bytes сырые байты всего multipart-тела.
     * @param ct заголовок Content-Type, содержащий маркер boundary.
     * @return отформатированная строка с детализацией по каждой части.
     */
    private fun formatMultipart(bytes: ByteArray, ct: String?): String {
        if (bytes.isEmpty()) {
            return BODY_EMPTY
        }

        // Если Pretty Print выключен — выводим Multipart как единый текстовый блок
        if (!logProps.prettyPrint) {
            return formatBody(bytes, ct)
        }

        val boundary = ct?.substringAfter(MARKER_BOUNDARY, STRING_EMPTY)
            ?.substringBefore(SEMICOLON)
            ?.replace(QUOTE, STRING_EMPTY)
            ?.trim() ?: return formatBody(bytes, ct)

        val delimiter = "$NEW_LINE$DOUBLE_DASH${boundary.trim()}"

        return runCatching {
            bytes.toString(Charsets.UTF_8)
                .split(delimiter)
                .filter { it.isNotBlank() && it != DOUBLE_DASH && it.contains(CONTENT_DISPOSITION) }
                .joinToString(NEW_LINE) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headerLines = lines.takeWhile { it.isNotBlank() }

                    val partHeaders = headerLines.associate {
                        it.substringBefore(COLON_SPACE) to it.substringAfter(COLON_SPACE)
                    }

                    val content = lines.drop(headerLines.size).dropWhile { it.isBlank() }.joinToString(NEW_LINE)
                    val contentBytes = content.toByteArray()

                    var name = PART_UNKNOWN
                    var fileName: String? = null
                    var partCt: String? = null

                    partHeaders.forEach { (k, v) ->
                        if (k.contains(CONTENT_DISPOSITION)) {
                            name = v.substringAfter(EQUALS_NAME, STRING_EMPTY).substringBefore(QUOTE)
                            if (v.contains(MARKER_FILENAME)) {
                                fileName = v.substringAfter(EQUALS_FILENAME).substringBefore(QUOTE)
                            }
                        } else if (k.equals(CONTENT_TYPE, ignoreCase = true)) {
                            partCt = v
                        }
                    }

                    val description = if (fileName != null) {
                        TEMPLATE_PART_FILE_INFO.format(name, fileName, contentBytes.size.toLong())
                    } else {
                        name
                    }

                    val headersInfo = if (partHeaders.isNotEmpty()) " | Headers: $partHeaders" else STRING_EMPTY

                    if (isBinaryContent(contentBytes, partCt, partHeaders, fileName)) {
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, BODY_BINARY)
                    } else {
                        val formattedContent = formatBody(contentBytes, partCt, partHeaders)
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, "Content: $formattedContent")
                    }
                }
        }.getOrElse { BODY_MULTIPART_RAW }
    }

    /**
     * Форматирует тело сообщения для вывода в лог.
     *
     * Логика работы:
     * 1. Проверка на пустое тело.
     * 2. Определение бинарного контента (MIME, расширение, байты).
     * 3. Если Pretty Print выключен — вывод "как есть".
     * 4. Если Pretty Print включен — попытка отформатировать JSON.
     * 5. Применение ограничений на размер (Truncate).
     */
    private fun formatBody(content: ByteArray, ct: String?, headers: Map<String, String>? = null): String {
        // 1. Быстрая проверка на пустое тело
        if (content.isEmpty()) {
            return BODY_EMPTY
        }

        if (content.contentEquals(BODY_TOO_LARGE.toByteArray(Charsets.UTF_8))) {
            return BODY_TOO_LARGE
        }

        // 2. Комплексная проверка на бинарные данные (наша финальная сигнатура)
        if (logProps.prettyPrint && !isMultipart(ct) && isBinaryContent(content, ct, headers)) {
            return BODY_BINARY
        }

        // 3. Формирование текстового представления
        val resultText = if (!logProps.prettyPrint) {
            // Режим Raw: просто переводим байты в строку
            content.toString(Charsets.UTF_8)
        } else {
            // Режим Pretty: пытаемся причесать JSON, если это он
            val isJson = ct?.contains(MediaType.APPLICATION_JSON_VALUE) == true || isJsonContent(content)
            if (isJson) {
                runCatching {
                    val tree = prettyMapper.readTree(content)
                    prettyMapper.writeValueAsString(tree)
                }.getOrElse { content.toString(Charsets.UTF_8) }
            } else {
                content.toString(Charsets.UTF_8)
            }
        }

        // 4. Проверка лимитов и обрезка (Truncate)
        return if (logProps.limitLogSize > 0 && resultText.length > logProps.limitLogSize) {
            val head = resultText.take(logProps.truncateChunkSize)
            val tail = resultText.takeLast(logProps.truncateChunkSize)
            val skipped = resultText.length - (head.length + tail.length)

            "$head\n$SUFFIX_TRUNCATED [Skipped $skipped characters]\n$tail"
        } else {
            resultText
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
     * Определяет, является ли контент бинарным на основе заголовков, имени файла и анализа байтов.
     *
     * Алгоритм работает по принципу «от транспортного уровня к контенту»:
     * 1. Проверяет транспортное сжатие (GZIP, Brotli, Deflate). Если данные сжаты — они бинарны.
     * 2. Проверяет явно текстовые MIME-типы (JSON, XML и т.д.). Если да — доверяем и считаем текстом.
     * 3. Анализирует подозрительные MIME-маркеры (картинки, архивы, офисные документы).
     * 4. Проверяет расширение файла (актуально для Multipart-частей или вложений).
     * 5. Финальная эвристика по байтам (поиск NULL-символов и непечатных знаков).
     *
     * @param bytes массив байтов для анализа.
     * @param headers карта заголовков (обычно [HttpHeaders.toSingleValueMap]).
     * @param contentType строка заголовка Content-Type.
     * @param fileName имя файла (из параметров парты или Content-Disposition).
     * @return true, если контент классифицирован как бинарный; false, если это текст.
     */
    private fun isBinaryContent(bytes: ByteArray, contentType: String? = null, headers: Map<String, String>? = null, fileName: String? = null): Boolean {

        // 1. Проверка на GZIP (приоритет над текстом)
        // Если данные сжаты (по заголовку или сигнатуре), они бинарны для логгера.
        val encoding = headers?.get(HttpHeaders.CONTENT_ENCODING)
        if (!encoding.isNullOrEmpty() && COMPRESSION_ENCODINGS.any { encoding.contains(it, ignoreCase = true) } || isGzipSignature(bytes)) {
            return true
        }
        // 2. ТЕКСТ: Если тип текстовый — доверяем и выводим как текст
        if (isAlwaysTextField(contentType)) {
            return false
        }

        // 3. MIME: Проверка по бинарным маркерам
        if (contentType?.lowercase()?.let { ct -> BINARY_MIME_MARKERS.any { ct.contains(it) } } == true) {
            return true
        }

        // 4. ФАЙЛ: Проверка по расширению
        // Берем либо переданный fileName, либо пытаемся вытащить его из Content-Type
        val effectiveFileName = fileName ?: extractFileNameFromContentType(contentType)
        if (effectiveFileName != null) {
            val ext = effectiveFileName.substringAfterLast('.', STRING_EMPTY).lowercase()
            if (ext.isNotEmpty() && ext !in EXTENSIONS_TEXT) {
                return true
            }
        }

        // 4. И только в последнюю очередь, если метаданных нет, смотрим на байты
        return !isText(bytes)
    }

    /**
     * Проверяет магические байты GZIP (0x1f, 0x8b) в начале массива.
     */
    private fun isGzipSignature(bytes: ByteArray): Boolean {
        if (bytes.size < 2) {
            return false
        }
        return (bytes[0].toInt() and 0xFF == 0x1F) && (bytes[1].toInt() and 0xFF == 0x8B)
    }

    /**
     * Пытается вытащить расширение, если оно спрятано в Content-Type
     * (иногда бывает: application/vnd.ms-excel; name="report.xlsx")
     */
    private fun extractFileNameFromContentType(ct: String?): String? {
        if (ct == null || !ct.contains(EQUALS_NAME)) {
            return null
        }

        return runCatching {
            ct.substringAfter(EQUALS_NAME)
                .substringBefore(QUOTE)
                .takeIf { it.isNotBlank() }
        }.getOrNull()
    }

    /**
     * Эвристически определяет, является ли массив байтов текстовым контентом.
     *
     * Алгоритм анализирует начало массива (первые [LIMIT_TEXT_CHECK_THRESHOLD] байт) на наличие
     * непечатных управляющих символов ASCII.
     *
     * Основные правила:
     * 1. Наличие NULL-байта (0x00) однозначно классифицирует контент как бинарный.
     * 2. Наличие управляющих символов (диапазон 0x00-0x1F), за исключением стандартных
     *    символов форматирования (табуляция, перенос строки, возврат каретки), считается
     *    признаком бинарных данных.
     * 3. Байты выше 0x7F (включая кириллицу UTF-8, иероглифы и эмодзи) считаются допустимыми,
     *    так как они являются частью многобайтовых текстовых кодировок.
     *
     * @param bytes Массив байтов для анализа.
     * @return true, если данные похожи на текст (JSON, XML, Plain Text);
     *         false, если обнаружены признаки бинарных данных (изображения, архивы и т.д.).
     */
    private fun isText(bytes: ByteArray): Boolean {
        if (bytes.isEmpty()) {
            return true
        }

        // Берем чуть больше данных для анализа
        val limit = minOf(bytes.size, LIMIT_TEXT_CHECK_THRESHOLD)

        for (i in 0 until limit) {
            val b = bytes[i].toInt() and 0xFF

            // NULL-байт — это 100% бинарник в контексте REST
            if (b == 0) {
                return false
            }

            // Управляющие символы ASCII (кроме таба и переносов)
            // Если их больше определенного порога, значит это не случайный символ, а бинарные данные
            if (b < 32 && b != 9 && b != 10 && b != 13) {
                // Можно добавить счетчик, но обычно даже одного такого байта
                // в начале JSON/XML быть не может.
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

        private const val STRING_EMPTY = ""
        private const val NEW_LINE = "\n"
        private const val COLON_SPACE = ": "
        private const val QUOTE = "\""
        private const val SLASH = "/"
        private const val SEMICOLON = ";"
        private const val DOUBLE_DASH = "--"

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_MULTIPART_RAW = "[MULTIPART RAW]"
        private const val BODY_LOG_DISABLED = "[BODY LOGGING DISABLED BY CLIENT]"
        private const val BODY_TOO_LARGE = "[BODY TOO LARGE TO LOG]"

        private val COMPRESSION_ENCODINGS = setOf("gzip", "br", "deflate")
        private val EXTENSIONS_TEXT = setOf(
            "txt", "json", "xml", "html", "csv", "yml", "yaml", // Базовые
            "log", "toml", "properties", "conf", "config",    // Конфиги и логи
            "md", "sql", "js", "css", "sh", "bat", "py"       // Скрипты и разметка
        )
        private val BINARY_MIME_MARKERS = setOf(
            "octet-stream", "image/", "video/", "audio/", "pdf", "zip",
            "vnd.ms-excel", "vnd.openxmlformats-officedocument", "application/msword"
        )
        const val ENCODING_GZIP = "gzip"

        private const val PART_UNKNOWN = "UNKNOWN"
        private const val TEMPLATE_PART_PREFIX = "  [PART] -> Name: %s%s | %s"
        private const val TEMPLATE_PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"

        private const val MARKER_BOUNDARY = "boundary="
        private const val MARKER_FILENAME = "filename="
        private const val MARKER_TEXT = "text"
        private const val MARKER_HTTP_PROTOCOL = "http"

        private const val EQUALS_NAME = "name=\""
        private const val EQUALS_FILENAME = "filename=\""

        private const val SUFFIX_TRUNCATED = "... [TRUNCATED]"
        private const val LIMIT_TEXT_CHECK_THRESHOLD = 512

        /** Байтовые константы для оптимизации производительности в циклах */
        private const val BYTE_SPACE = 32.toByte()
        private const val BYTE_JSON_OBJECT = '{'.code.toByte()
        private const val BYTE_JSON_ARRAY = '['.code.toByte()

        const val ATTR_SKIP_LOGGING = "client.skip.body.logging"
        const val ATTR_EXT_RQ_ID = "client.ext.request.id"
        const val MDC_NEW_EXT_RQ_ID = "newExtRqId"
    }
}
