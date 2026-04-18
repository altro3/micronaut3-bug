package com.micronaut.bug.client

import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.databind.json.JsonMapper
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpRequest
import org.springframework.http.MediaType
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.text.Charsets.UTF_8

/**
 * Фильтр для логирования запросов во внешние системы
 */
class LoggingRequestInterceptorOld(
    private val httpClientProperties: HttpClientProperties,
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}
    /**
     * Маппер для pretty-print вывода json'ов логах.
     */
    private val objectMapper = JsonMapper.builder()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .build()
    private val baseUrl: String = if (httpClientProperties.logFullUrl) httpClientProperties.url.toString() else ""

    override fun intercept(
        request: HttpRequest,
        requestBody: ByteArray,
        execution: ClientHttpRequestExecution,
    ): ClientHttpResponse {
        val response: ClientHttpResponse
        if (log.isDebugEnabled()) {
            logRequest(request, requestBody)
            response = execution.execute(request, requestBody)
            logResponse(response)
        } else {
            response = execution.execute(request, requestBody)
            if (response.statusCode.isError) {
                logError(request, requestBody, response)
            }
        }
        return response
    }

    private fun logRequest(rq: HttpRequest, body: ByteArray) {
        val contentType = rq.headers.getFirst(HttpHeaders.CONTENT_TYPE) ?: ""
        val contentDisposition = rq.headers.getFirst(HttpHeaders.CONTENT_DISPOSITION)
        log.debug {
            """
============================== Request begin ======================
Request URI: ${rq.method} ${baseUrl + rq.uri}
Request Headers: ${rq.headers}
Request Body: ${getBody(contentType, contentDisposition, body) ?: EMPTY_REQUEST_BODY}
============================== Request end ========================"""
        }
    }

    private fun logResponse(rs: ClientHttpResponse) {

        val body = try {
            rs.body.readAllBytes()
        } catch (e: Exception) {
            log.debug { "Can't read response body" }
            EMPTY_BYTE_ARRAY
        }
        val contentType = rs.headers.getFirst(HttpHeaders.CONTENT_TYPE) ?: ""
        val contentDisposition = rs.headers.getFirst(HttpHeaders.CONTENT_DISPOSITION)

        log.debug {
            """
============================== Response begin ======================
Response Status: ${rs.statusCode} ${rs.statusText}
Response Headers: ${rs.headers}
Response Body: ${getBody(contentType, contentDisposition, body) ?: EMPTY_RESPONSE_BODY}
============================== Response end ========================"""
        }
    }

    private fun logError(
        rq: HttpRequest,
        rqBody: ByteArray,
        rs: ClientHttpResponse,
    ) {
        val rqContentType = rq.headers.getFirst(HttpHeaders.CONTENT_TYPE) ?: ""
        val rqContentDisposition = rq.headers.getFirst(HttpHeaders.CONTENT_DISPOSITION)

        val rsContentType = rs.headers.getFirst(HttpHeaders.CONTENT_TYPE) ?: ""
        val rsContentDisposition = rs.headers.getFirst(HttpHeaders.CONTENT_DISPOSITION)
        val rsBody = try {
            rs.body.readAllBytes()
        } catch (e: Exception) {
            log.warn { "Can't read response body" }
            EMPTY_BYTE_ARRAY
        }

        log.warn {
            """
============================== Failed HTTP-request begin ======================
Request URI: ${rq.method} ${baseUrl + rq.uri}
Request Headers: ${rq.headers}
Request Body: ${getBody(rqContentType, rqContentDisposition, rqBody) ?: EMPTY_REQUEST_BODY}
Response Status: ${rs.statusCode} ${rs.statusText}
Response Headers: ${rs.headers}
Response Body: ${getBody(rsContentType, rsContentDisposition, rsBody) ?: EMPTY_RESPONSE_BODY}
============================== Failed HTTP-request end ========================"""
        }
    }

    private fun getBody(contentType: String, contentDisposition: String?, body: ByteArray): String? {
        return when {
            body.isEmpty() -> null
            contentType.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE, true) -> getMultipartBody(contentType, body)
            contentType == MediaType.APPLICATION_OCTET_STREAM_VALUE || !contentDisposition.isNullOrEmpty() -> BODY_BINARY_FILE
            contentType.startsWith(MediaType.APPLICATION_JSON_VALUE, true) ->
                try {
                    "\n" + objectMapper.writeValueAsString(objectMapper.readTree(body))
                } catch (e: Exception) {
                    String(body)
                }
            else -> String(body)
        }
    }

    private fun getMultipartBody(contentType: String?, body: ByteArray): String {

        val resultBytes = ByteArrayOutputStream()

        if (!httpClientProperties.logMultipartBody) {
            resultBytes.write(DISABLED_LOG)
            resultBytes.close()
            return resultBytes.toString(UTF_8)
        }
        val multipartBoundary = contentType?.startsWith(MediaType.MULTIPART_FORM_DATA_VALUE, true)
            ?.let {
                REGEX_BOUNDARY.find(contentType)?.groups?.get(PARAM_MULTIPART_DATA_BOUNDARY)
            }?.value

        val requestByteReader = ByteArrayInputStream(body).bufferedReader()

        var isFile = false
        while (requestByteReader.ready()) {
            do {
                val line = requestByteReader.readLine()
                when {
                    line.startsWith(HttpHeaders.CONTENT_DISPOSITION, true) -> {
                        isFile = line.contains(PARAM_CONTENT_DISPOSITION_FILENAME)
                        resultBytes.write(line.plus('\n').toByteArray())
                        if (isFile) {
                            resultBytes.write(SKIP_BINARY_DATA)
                        }
                        continue
                    }
                    else -> {
                        if (isFile) {
                            continue
                        }
                        resultBytes.write(line.plus('\n').toByteArray())
                    }
                }
            } while (multipartBoundary?.let { line.endsWith(it) } == false && requestByteReader.ready())
            isFile = false
        }

        requestByteReader.close()
        resultBytes.close()
        return resultBytes.toString(UTF_8)
    }

    companion object {

        private val DISABLED_LOG = "<Disabled log for multipart/form-data>\n".toByteArray()
        private val SKIP_BINARY_DATA = "<Skip binary data>\n".toByteArray()
        private val REGEX_BOUNDARY = Regex("boundary=(?<boundary>.*)$")
        private val EMPTY_BYTE_ARRAY = ByteArray(0)

        private const val EMPTY_REQUEST_BODY = "<Empty request body>"
        private const val EMPTY_RESPONSE_BODY = "<Empty response body>"
        private const val PARAM_CONTENT_DISPOSITION_FILENAME = "filename"
        private const val PARAM_MULTIPART_DATA_BOUNDARY = "boundary"
        private const val BODY_BINARY_FILE = "Binary / text file"
    }
}
