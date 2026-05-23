package com.altro.service1.service.integration.extservice

import com.altro.common.client.DefaultHttpClient
import com.altro.service1.service.integration.extservice.api.MyDataRequest
import com.altro.service1.service.integration.extservice.api.MyDataResponse
import com.altro.service1.service.integration.extservice.api.MyDto
import com.altro.service1.service.integration.extservice.config.ExtServiceProperties
import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.core.ParameterizedTypeReference
import org.springframework.core.io.Resource
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.util.LinkedMultiValueMap
import org.springframework.util.MultiValueMap
import java.io.ByteArrayOutputStream

@Service
class ExtServiceClient(
    props: ExtServiceProperties,
    @Qualifier("extServiceHttpClient")
    val httpClient: DefaultHttpClient,
    private val objectMapper: ObjectMapper,
) {

    private val endpoints = props.endpoints

    // 1. Гет: пустое тело запроса, пустое тело ответа (204 No Content или просто игнор тела)
    fun ping() {
        httpClient.sendRq(endpoints.ping).toBodilessEntity()
    }

    // 2. Пост: передача файла и получение файла
    fun processFile(fileResource: Resource): ByteArray? {
        return httpClient.sendRq(
            endpoint = endpoints.processFile,
            rqBody = fileResource,
            headers = mapOf(HttpHeaders.CONTENT_TYPE to MediaType.APPLICATION_OCTET_STREAM_VALUE)
        ).body(ByteArray::class.java)
    }

    // 3. Пост: JSON -> JSON
    fun updateData(request: MyDataRequest): MyDataResponse? {
        return httpClient.sendRq(
            endpoint = endpoints.updateData,
            responseClass = MyDataResponse::class.java,
            rqBody = request
        )
    }

    // 4. Пост: Сложный Multipart + Path Variable + Query Parameter
    fun sendComplexMultipart(
        id: String,
        queryTag: String,
        jsonWithCt: MyDto,
        jsonNoCt: MyDto,
        binaryFile: Resource,
        textFile: String
    ): MultiValueMap<String, Any>? { // Предположим, сервис возвращает multipart
        // Собираем Multipart-тело
        val body: MultiValueMap<String, Any> = LinkedMultiValueMap()

        // JSON с указанием Content-Type
        val jsonPart = HttpEntity(jsonWithCt, HttpHeaders().apply { contentType = MediaType.APPLICATION_JSON })
        body.add("jsonWithCt", jsonPart)

        // JSON без явного указания (будет использован дефолт конвертера)
        body.add("jsonNoCt", jsonNoCt)

        // Бинарный файл
        body.add("binaryFile", binaryFile)

        // Текстовый файл (как часть формы)
        body.add("textFile", textFile)

        return httpClient.sendRq(
            endpoint = endpoints.complexMultipart,
            responseType = object : ParameterizedTypeReference<MultiValueMap<String, Any>>() {},
            rqBody = body,
            pathVars = mapOf("id" to id),
            queryParams = mapOf("tag" to queryTag),
            headers = mapOf(HttpHeaders.CONTENT_TYPE to MediaType.MULTIPART_FORM_DATA_VALUE)
        )
    }

    fun testAutoGzipTransfer(request: MyDataRequest): MyDataResponse? {
        return httpClient.sendRq(
            // Предположим, добавили gzip в Endpoints проперти
            path = "/v1/data/gzip",
            method = HttpMethod.POST,
            rqBody = request,
            responseClass = MyDataResponse::class.java,
            // Мы можем добавить заголовок вручную, чтобы Netty точно сжал тело запроса
            headers = mapOf(HttpHeaders.CONTENT_ENCODING to "gzip"),
        )
    }

    fun testGzipTransfer(request: MyDataRequest) =
        httpClient.sendRq(
            // Предположим, добавили gzip в Endpoints проперти
            path = "/v1/data/manual-gzip",
            method = HttpMethod.POST,
            rqBody = compress(objectMapper.writeValueAsBytes(request)),
            responseClass = ByteArray::class.java,
            headers = mapOf(
                HttpHeaders.CONTENT_ENCODING to "gzip",
                HttpHeaders.CONTENT_TYPE to MediaType.APPLICATION_OCTET_STREAM_VALUE,
                HttpHeaders.ACCEPT to MediaType.APPLICATION_OCTET_STREAM_VALUE,
            ),
        )

    fun testPlainToGzip(request: MyDataRequest): MyDataResponse? {
        return httpClient.sendRq(
            path = "/v1/data/plain-to-gzip",
            method = HttpMethod.POST,
            rqBody = request,
            responseClass = MyDataResponse::class.java
            // Заголовок Content-Encoding не передаем, GzipRequestInterceptor не сработает
        )
    }

    fun testError(request: MyDataRequest): MyDataResponse? {
        return httpClient.sendRq(
            path = "/unknown",
            method = HttpMethod.POST,
            rqBody = request,
            responseClass = MyDataResponse::class.java
            // Заголовок Content-Encoding не передаем, GzipRequestInterceptor не сработает
        )
    }

    fun testHugeData(data: ByteArray): ByteArray? {
        return httpClient.sendRq(
            path = "/v1/data/huge-proxy",
            method = HttpMethod.POST,
            rqBody = data,
            responseClass = ByteArray::class.java,
        )
    }

    private fun compress(body: ByteArray): ByteArray {
        val baos = ByteArrayOutputStream()
        // Сначала полностью завершаем работу со стримом
        java.util.zip.GZIPOutputStream(baos).use { gzip ->
            gzip.write(body)
            gzip.finish() // Явно завершаем формирование архива
        }
        // Только ПОСЛЕ закрытия/финиша берем байты
        return baos.toByteArray()
    }
}
