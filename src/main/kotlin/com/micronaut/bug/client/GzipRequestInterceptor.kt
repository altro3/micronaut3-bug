package com.micronaut.bug.client

import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.ATTR_EXT_RQ_ID
import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.ENCODING_GZIP
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse
import java.io.ByteArrayOutputStream
import java.util.zip.GZIPOutputStream

/**
 * Интерцептор для автоматического сжатия тела исходящих запросов.
 *
 * Работает в связке с настройкой [HttpClientProperties.compress].
 * Сжимает данные только если:
 * 1. Тело запроса не пустое.
 * 2. Установлен заголовок Content-Encoding: gzip.
 *
 * Важно: в цепочке интерцепторов должен стоять ПОСЛЕ логгера,
 * чтобы в логи попадал читаемый текст, а в сеть уходили сжатые байты.
 */
class GzipRequestInterceptor : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}

    override fun intercept(
        rq: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {
        val isGzipRequested = rq.headers.getFirst(HttpHeaders.CONTENT_ENCODING)
            ?.contains(ENCODING_GZIP, true) == true

        if (isGzipRequested && body.isNotEmpty()) {
            val compressedBody = compress(body)

            // Извлекаем наш ID из атрибутов
            val extRqId = rq.attributes[ATTR_EXT_RQ_ID] as? String ?: EXT_REQ_ID_UNKNOWN

            log.debug {
                val original = body.size.toLong()
                val compressed = compressedBody.size.toLong()
                val diff = compressed - original

                // Вычисляем процент изменения через Double для точности
                val percentChange = (diff.toDouble() / original * 100).toInt()

                // Формируем наглядный индикатор изменения (например: -450 B или +12 B)
                val diffFormatted = if (diff > 0) "+$diff B" else "$diff B"
                val percentFormatted = if (percentChange > 0) "+$percentChange%" else "$percentChange%"

                "GZIP applied [extRqId: $extRqId]. Size: $original -> $compressed bytes | Diff: $diffFormatted ($percentFormatted)"
            }
            return execution.execute(rq, compressedBody)
        }

        return execution.execute(rq, body)
    }

    /**
     * Сжимает массив байтов по алгоритму GZIP.
     */
    private fun compress(body: ByteArray): ByteArray {
        val baos = ByteArrayOutputStream()
        GZIPOutputStream(baos).use { it.write(body) }
        return baos.toByteArray()
    }

    companion object {

        const val EXT_REQ_ID_UNKNOWN = "unknown"
    }
}
