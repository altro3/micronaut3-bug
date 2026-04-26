package com.micronaut.bug.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.micronaut.bug.client.HttpClientConst.HEADER_EXT_RQ_ID
import com.micronaut.bug.config.ServerLoggingFilter.Companion.LIMIT_TEXT_CHECK_THRESHOLD
import com.micronaut.bug.config.log.LogProperties
import com.micronaut.bug.util.TraceIdGenerator
import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.ReadListener
import jakarta.servlet.ServletInputStream
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletRequestWrapper
import jakarta.servlet.http.HttpServletResponse
import jakarta.servlet.http.Part
import org.slf4j.MDC
import org.springframework.http.HttpHeaders.CONTENT_DISPOSITION
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.util.ClassUtils
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingResponseWrapper
import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.InputStreamReader

/**
 * Фильтр для детального логирования входящих HTTP-запросов и ответов на стороне сервера.
 * Поддерживает защиту от OOM, разбор Multipart и Pretty Print для JSON.
 */
class ServerLoggingFilter(
    objectMapper: ObjectMapper,
    private val logProps: LogProperties,
    private val appName: String,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    /**
     * Mapper для красивого вывода JSON.
     * Копируем основной, чтобы не менять глобальные настройки сериализации.
     */
    private val prettyMapper: ObjectMapper = objectMapper.copy()
        .enable(SerializationFeature.INDENT_OUTPUT)
    private val withActuator: Boolean = ClassUtils.isPresent("org.springframework.boot.actuate.autoconfigure.endpoint.web.WebEndpointProperties", null)

    override fun doFilterInternal(
        rq: HttpServletRequest,
        rs: HttpServletResponse,
        chain: FilterChain,
    ) {
        // Извлекаем или генерируем ID запроса для сквозной трассировки в MDC
        val rqId = rq.getHeader(HEADER_X_RQ_ID)?.takeIf { it.isNotBlank() } ?: genTraceId()
        MDC.put(MDC_RQ_ID, rqId)
        val sender = rq.getHeader(HEADER_X_SENDER) ?: "USER"
        MDC.put(MDC_CLIENT, sender)
        MDC.put(MDC_SERVER, appName)
        val extRqId = rq.getHeader(HEADER_EXT_RQ_ID)
        if (extRqId != null) {
            MDC.put(MDC_EXT_RQ_ID, extRqId)
        }

        val startTime = System.currentTimeMillis()

        try {

            // Пропускаем Actuator-эндпоинты, если это указано в настройках
            if (withActuator && logProps.skipActuator && rq.requestURI.startsWith(rq.contextPath + PATH_ACTUATOR)) {
                chain.doFilter(rq, rs)
                return
            }

            val isDebugProvider = log.isDebugEnabled() && logProps.enabledControllerLogging

            // Оборачиваем запрос: кэшируем тело, чтобы прочитать его для лога и оставить доступным для контроллера
            val currentRq = wrapRequest(rq)

            // Сразу формируем строку запроса, но не печатаем её
            val requestLogData by lazy { getRequestLogString(currentRq) }

            // Оборачиваем ответ для кэширования исходящего потока
            val rsWrapper = ContentCachingResponseWrapper(rs)
            rsWrapper.bufferSize = logProps.maxPayloadSize.toBytes().toInt()

            try {
                // Если включен обычный дебаг-логинг — печатаем запрос сразу
                if (isDebugProvider) {
                    log.debug { requestLogData }
                }

                chain.doFilter(currentRq, rsWrapper)
            } finally {
                val duration = System.currentTimeMillis() - startTime
                val status = rsWrapper.status
                val isError = status >= 400
                val isFullBodyCached = rsWrapper.contentSize < logProps.maxPayloadSize.toBytes()

                val rsBodyBytes = when {
                    rsWrapper.contentSize == 0 -> BODY_EMPTY.toByteArray(Charsets.UTF_8)
                    !isFullBodyCached -> BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)
                    else -> rsWrapper.contentAsByteArray
                }

                if (isError && !isDebugProvider) {
                    // ПРИ ОШИБКЕ: логируем и запрос, и ответ на уровне ERROR
                    log.error { "Service failure detected!\n$requestLogData\n${getResponseLogString(currentRq, rsWrapper, rsBodyBytes, duration)}" }
                } else if (isDebugProvider) {
                    // В штатном режиме: логируем только ответ в DEBUG
                    log.debug { getResponseLogString(currentRq, rsWrapper, rsBodyBytes, duration) }
                }
                // Важно: копируем кэшированное тело ответа обратно в реальный поток
                rsWrapper.copyBodyToResponse()
            }
        } finally {
            MDC.clear()
        }
    }

    private fun getResponseLogString(
        rq: HttpServletRequest,
        rs: ContentCachingResponseWrapper,
        body: ByteArray,
        duration: Long
    ): String {
        val statusInt = rs.status
        val httpStatus = HttpStatus.resolve(statusInt)
        val statusMessage = httpStatus?.reasonPhrase ?: if (statusInt == 0) STATUS_UNDEFINED else STATUS_UNKNOWN

        val headers = getResponseHeaders(rs)

        // formatBody уже умеет обрабатывать константу BODY_TOO_LARGE
        val bodyText = when {
            isMultipart(rs.contentType) && !body.contentEquals(BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)) ->
                formatMultipartResponse(body, rs.contentType)

            else -> formatBody(body, rs.contentType, headers)
        }

        return """
            |
            |================== Service response ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |Status: $statusInt $statusMessage
            |Duration: ${duration}ms
            |Headers: $headers
            |Body:
            |$bodyText
            |================== /Service response ==================
        """.trimMargin()
    }

    /**
     * Оборачивает входящий запрос для обеспечения возможности повторного чтения тела.
     *
     * Если размер контента превышает лимит, запрос остается оригинальным
     * и его тело НЕ будет закешировано. Это предотвращает OutOfMemoryError при
     * передаче больших файлов.
     *
     * @param rq оригинальный [HttpServletRequest].
     * @return обернутый или оригинальный запрос.
     */
    private fun wrapRequest(rq: HttpServletRequest): HttpServletRequest {
        return if (isMultipart(rq.contentType)) {
            MultipartTypeWrapper(rq)
        } else {
            val contentLength = rq.contentLengthLong
            // Если размер тела больше лимита логирования (например, > 1МБ),
            // мы НЕ кэшируем его, а пробрасываем оригинальный поток.
            if (contentLength > logProps.maxPayloadSize.toBytes()) {
                rq
            } else {
                val bytes = rq.inputStream.readAllBytes()
                CachedBodyRequestWrapper(rq, bytes)
            }
        }
    }

    /**
     * Формирует строковое представление входящего запроса.
     */
    private fun getRequestLogString(rq: HttpServletRequest): String {
        val headers = getHeadersMap(rq)
        val bodyResult = when (rq) {
            is CachedBodyRequestWrapper -> {
                if (rq.body.isEmpty()) {
                    BODY_EMPTY
                } else {
                    formatBody(rq.body, rq.contentType, headers)
                }
            }

            is MultipartTypeWrapper -> getMultipartBody(rq)
            else -> BODY_TOO_LARGE
        }

        return """
            |
            |================== Service request ==================
            |URI: ${rq.method} ${getFullUri(rq)}
            |Headers: $headers
            |Body:
            |$bodyResult
            |================== /Service request ==================
        """.trimMargin()
    }

    /**
     * Разбирает Multipart запрос на отдельные части и форматирует их.
     */
    private fun getMultipartBody(rq: MultipartTypeWrapper): String =
        runCatching {
            rq.parts.joinToString(NEW_LINE) { part ->
                val observedPart = part as ObservedPart
                val bytes = observedPart.getContentBytes()

                // Собираем хедеры текущей части
                val partHeaders = observedPart.headerNames.associateWith { observedPart.getHeader(it) }

                val description = part.submittedFileName?.let {
                    TEMPLATE_PART_FILE_INFO.format(part.name, it, part.size)
                } ?: part.name

                // Передаем хедеры в метод обработки контента
                processPartContent(bytes, part.contentType, description, partHeaders)
            }.ifEmpty { BODY_EMPTY }
        }.getOrElse { "[MULTIPART ERROR: ${it.message}]" }

    /**
     * Обрабатывает контент конкретной части Multipart сообщения.
     */
    private fun processPartContent(
        bytes: ByteArray,
        ct: String?,
        description: String,
        headers: Map<String, String> = emptyMap()
    ): String {
        val headersString = if (headers.isNotEmpty()) " | Headers: $headers" else STRING_EMPTY
        val content = formatBody(bytes, ct, headers)
        val prefix = if (content == BODY_BINARY) BODY_BINARY else "Content: $content"
        return TEMPLATE_PART_PREFIX.format(description, headersString, prefix)
    }

    /**
     * Ручной разбор Multipart-ответа (используется редко, для симметрии с клиентом).
     */
    private fun formatMultipartResponse(bytes: ByteArray, contentType: String?): String {
        if (bytes.isEmpty()) {
            return BODY_EMPTY
        }

        val boundary = contentType?.substringAfter(MARKER_BOUNDARY, STRING_EMPTY)
            ?.substringBefore(SEMICOLON)
            ?.replace(QUOTE, STRING_EMPTY)
            ?.trim() ?: return BODY_MULTIPART_RS

        val delimiter = "$NEW_LINE$DOUBLE_DASH${boundary.trim()}"

        return runCatching {
            bytes.toString(Charsets.UTF_8).split(delimiter)
                .filter { it.isNotBlank() && it != DOUBLE_DASH && it.contains(CONTENT_DISPOSITION) }
                .joinToString(NEW_LINE) { partRaw ->
                    val lines = partRaw.trim().lines()
                    val headerLines = lines.takeWhile { it.isNotBlank() }

                    // Сбор хедеров парта из строк
                    val partHeaders = headerLines.associate {
                        it.substringBefore(COLON_SPACE) to it.substringAfter(COLON_SPACE)
                    }

                    val content = lines.dropWhile { it.isNotBlank() }.drop(1).joinToString(NEW_LINE)

                    var name: String? = null
                    var fileName: String? = null
                    var partCt: String? = null

                    // Извлекаем метаданные из собранных хедеров
                    partHeaders.forEach { (k, v) ->
                        when {
                            k.contains(CONTENT_DISPOSITION) -> {
                                name = v.substringAfter(EQUALS_NAME, STRING_EMPTY).substringBefore(QUOTE)
                                fileName = if (v.contains(MARKER_FILENAME)) v.substringAfter(EQUALS_FILENAME).substringBefore(QUOTE) else null
                            }

                            k.equals(CONTENT_TYPE, ignoreCase = true) -> partCt = v
                        }
                    }

                    val finalName = name ?: PART_UNKNOWN
                    val contentBytes = content.toByteArray()
                    val description = if (fileName != null) {
                        TEMPLATE_PART_FILE_INFO.format(finalName, fileName, contentBytes.size.toLong())
                    } else {
                        finalName
                    }

                    val headersInfo = if (partHeaders.isNotEmpty()) " | Headers: $partHeaders" else STRING_EMPTY
                    if (isBinaryContent(contentBytes, partHeaders, partCt, finalName)) {
                        TEMPLATE_PART_PREFIX.format(description, headersInfo, BODY_BINARY)
                    } else {
                        processPartContent(contentBytes, partCt, description, partHeaders)
                    }
                }
        }.getOrElse { BODY_MULTIPART_RS }
    }

    /**
     * Основной метод форматирования тела (JSON Pretty Print, Raw или Binary).
     */
    private fun formatBody(content: ByteArray, contentType: String?, headers: Map<String, String>? = null): String {
        if (content.isEmpty() || content.contentEquals(BODY_EMPTY.toByteArray())) {
            return BODY_EMPTY
        }
        if (content.contentEquals(BODY_TOO_LARGE.toByteArray())) {
            return BODY_TOO_LARGE
        }
        // Умная проверка на бинарные данные, включая GZIP и расширения
        if (logProps.prettyPrint && !isMultipart(contentType) && isBinaryContent(content, headers, contentType)) {
            return BODY_BINARY
        }

        val resultText = if (logProps.prettyPrint) {
            formatIfJson(content, contentType)
        } else {
            String(content, Charsets.UTF_8)
        }

        // Обрезка по принципу "голова...хвост"
        return if (logProps.limitLogSize > 0 && resultText.length > logProps.limitLogSize) {
            val head = resultText.take(logProps.truncateChunkSize)
            val tail = resultText.takeLast(logProps.truncateChunkSize)
            "$head\n$SUFFIX_TRUNCATED [Skipped ${resultText.length - 2 * logProps.truncateChunkSize} chars]\n$tail"
        } else {
            resultText
        }
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

    private fun getHeadersMap(rq: HttpServletRequest): Map<String, String> =
        rq.headerNames.asSequence().associateWith { rq.getHeader(it) }

    private fun getResponseHeaders(rs: HttpServletResponse): Map<String, String> =
        rs.headerNames.associateWith { rs.getHeader(it) }

    private class CachedBodyRequestWrapper(
        rq: HttpServletRequest,
        val body: ByteArray,
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
        const val HEADER_X_RQ_ID = "x-rq-id"
        const val HEADER_X_SENDER = "x-sender"
        const val MDC_RQ_ID = "rqId"
        const val MDC_CLIENT = "client"
        const val MDC_SERVER = "server"
        const val MDC_EXT_RQ_ID = "extRqId"

        private const val PATH_ACTUATOR = "/actuator"

        private const val STRING_EMPTY = ""
        private const val QUOTE = "\""
        private const val NEW_LINE = "\n"
        private const val COLON_SPACE = ": "
        private const val SEMICOLON = ";"
        private const val DOUBLE_DASH = "--"

        private const val BODY_EMPTY = "[EMPTY]"
        private const val BODY_BINARY = "[BINARY DATA]"
        private const val BODY_MULTIPART_RS = "[MULTIPART RAW DISABLED]"
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

        private const val PART_UNKNOWN = "UNKNOWN"

        private const val STATUS_UNDEFINED = "UNDEFINED"
        private const val STATUS_UNKNOWN = "UNKNOWN"

        private const val TEMPLATE_PART_PREFIX = "  [PART] -> Name: %s%s | %s"
        private const val TEMPLATE_PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"

        private const val SUFFIX_TRUNCATED = "... [TRUNCATED]"

        private const val MARKER_BOUNDARY = "boundary="
        private const val MARKER_FILENAME = "filename="
        private const val MARKER_TEXT = "text"

        private const val EQUALS_NAME = "name=\""
        private const val EQUALS_FILENAME = "filename=\""

        private const val LIMIT_TEXT_CHECK_THRESHOLD = 512

        private const val BYTE_JSON_OBJECT = '{'.code.toByte()
        private const val BYTE_JSON_ARRAY = '['.code.toByte()

        private fun isJsonContent(bytes: ByteArray): Boolean {
            // Ищем первый не пробельный символ
            for (b in bytes) {
                if (b <= 32) {
                    continue
                }
                return b == BYTE_JSON_OBJECT || b == BYTE_JSON_ARRAY
            }
            return false
        }

        /**
         * Определяет, является ли контент бинарным (GZIP, MIME, расширение или байты).
         * Синхронизировано с логикой клиентского интерцептора.
         */
        private fun isBinaryContent(bytes: ByteArray, headers: Map<String, String>? = null, contentType: String? = null, fileName: String? = null): Boolean {
            // 1. Проверка на сжатие (защита от кракозябр)
            val encoding = headers?.get(org.springframework.http.HttpHeaders.CONTENT_ENCODING)
            if (!encoding.isNullOrEmpty() && COMPRESSION_ENCODINGS.any { encoding.contains(it, ignoreCase = true) } || isGzipSignature(bytes)) {
                return true
            }

            if (isAlwaysTextField(contentType)) {
                return false
            }

            // 2. Проверка по MIME-типам
            if (contentType?.lowercase()?.let { ct -> BINARY_MIME_MARKERS.any { ct.contains(it) } } == true) {
                return true
            }

            // 3. Проверка по расширению файла
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
            TraceIdGenerator.generate()
    }
}
