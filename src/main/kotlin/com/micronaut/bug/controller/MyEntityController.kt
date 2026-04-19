package com.micronaut.bug.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.config.SecurityContext
import com.micronaut.bug.config.User
import com.micronaut.bug.service.BusinessService
import com.micronaut.bug.service.integration.extservice.ExtServiceClient
import com.micronaut.bug.service.integration.extservice.api.MyDataRequest
import com.micronaut.bug.service.integration.extservice.api.MyDto
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.io.ByteArrayResource
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.util.LinkedMultiValueMap
import org.springframework.util.MultiValueMap
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.ResponseStatus
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
class MyEntityController(
    private val businessService: BusinessService,
    private val objectMapper: ObjectMapper,
    private val extServiceClient: ExtServiceClient,
) {

    private val log = KotlinLogging.logger {}

    @PostMapping(
        "/testBinary",
        consumes = [MediaType.APPLICATION_OCTET_STREAM_VALUE],
        produces = [MediaType.APPLICATION_OCTET_STREAM_VALUE]
    )
    fun testBinary(@RequestBody bytes: ByteArray): ByteArray {
        log.info { "Received binary data, size: ${bytes.size}" }

        return bytes
    }

    @GetMapping(
        "/testJson",
        produces = [MediaType.APPLICATION_JSON_VALUE]
    )
    fun testJson(): MyData {
        log.info { "GET request for JSON received" }

        return MyData(name = "Test", secondName = "User", age = 25)
    }

    @PostMapping("/testJson2")
    fun testEmpty(@RequestBody body: MyData): MyData =
        body

    @PostMapping("/testEmpty")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    fun testEmpty() {
        log.info { "Empty request/response endpoint called" }
    }

    @PostMapping(
        "/testMultipart/{pathVar}",
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.MULTIPART_FORM_DATA_VALUE]
    )
    fun testMultipart(
        @PathVariable pathVar: String,
        @RequestParam queryVar: String,
        @RequestPart(required = false) data: String?,
        @RequestPart(required = false) file: MultipartFile?,
        @RequestPart(required = false) binaryFile: MultipartFile? // Новая необязательная бинарная парта
    ): MultiValueMap<String, Any> {
        val dataParsed = data?.let {
            objectMapper.readValue(it, MyData::class.java)
        }

        log.info { "body: ${data ?: "null"}" }
        log.info { "dataParsed: $dataParsed" }
        log.info { "pathVar: $pathVar" }
        log.info { "queryVar: $queryVar" }
        log.info { "binaryFile received: ${binaryFile?.originalFilename ?: "none"}, size: ${binaryFile?.size ?: 0}" }

        val currentUser = SecurityContext.getUser()
        val processedUser = businessService.processOrder(currentUser)

        val resultData = (dataParsed ?: MyData()).apply {
            user = processedUser
        }

        val responseMap = LinkedMultiValueMap<String, Any>()

        val dataHeaders = HttpHeaders().apply { contentType = MediaType.APPLICATION_JSON }
        responseMap.add(PART_RESPONSE_DATA, HttpEntity(resultData, dataHeaders))

        val infoBytes = "Received file: ${file?.originalFilename ?: "none"}".toByteArray()
        val infoResource = object : ByteArrayResource(infoBytes) {
            override fun getFilename(): String = INFO_FILENAME
        }
        val fileHeaders = HttpHeaders().apply { contentType = MediaType.TEXT_PLAIN }
        responseMap.add(PART_RESPONSE_FILE_INFO, HttpEntity(infoResource, fileHeaders))

        val binaryBytes = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01)
        val binaryResource = object : ByteArrayResource(binaryBytes) {
            override fun getFilename(): String = BINARY_FILENAME
        }
        val binaryHeaders = HttpHeaders().apply { contentType = MediaType.APPLICATION_OCTET_STREAM }
        responseMap.add(PART_RESPONSE_BINARY, HttpEntity(binaryResource, binaryHeaders))

        return responseMap
    }

    @GetMapping("/run-all")
    fun runAllTests(): String {
        log.info { "Starting sequential integration tests..." }

        // 1. Тест Ping (GET без тела)
        log.info { "--- Test 1: Ping ---" }
        extServiceClient.ping()

        // 2. Тест JSON (POST JSON -> JSON)
        log.info { "--- Test 2: Update Data (JSON) ---" }
        val updateRs = extServiceClient.updateData(MyDataRequest(name = "Test item"))
        log.info { "Update result: $updateRs" }

        // 3. Тест Process File (POST Binary -> Binary)
        log.info { "--- Test 3: Process File (Binary) ---" }

        // Создаем реальные бинарные данные, начинающиеся с NULL-байта
        val binaryContent = byteArrayOf(1, 2, 3, 4, 5, 0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte())

        val binaryFile = object : ByteArrayResource(binaryContent) {
            override fun getFilename() = "actual-binary-data.bin"
        }

        val fileRs = extServiceClient.processFile(binaryFile)

        log.info { "Binary request sent. Received response size: ${fileRs?.size ?: 0} bytes" }

        // 4. Тест Complex Multipart (Path Var + Query + Different Parts)
        log.info { "--- Test 4: Complex Multipart ---" }
        val complexRs = extServiceClient.sendComplexMultipart(
            id = "RQ-777",
            queryTag = "test-tag",
            jsonWithCt = MyDto("key", "val"),
            jsonNoCt = MyDto("k", "v"),
            binaryFile = object : ByteArrayResource(byteArrayOf(1, 2, 3)) { override fun getFilename() = "req.bin" },
            textFile = "hello"
        )

        if (complexRs != null) {
            log.info { "--- Multipart Response Verification ---" }

            // 1. Проверка JSON части (status)
            val statusEntity = complexRs.getFirst("status") as? HttpEntity<ByteArray>
            val statusJson = statusEntity?.body?.let { String(it) }
            log.info { "Part 'status': content=$statusJson, contentType=${statusEntity?.headers?.contentType}" }

            // 2. Проверка текстового файла (report)
            val reportEntity = complexRs.getFirst("report") as? HttpEntity<ByteArray>
            val reportText = reportEntity?.body?.let { String(it) }
            log.info { "Part 'report': filename=summary.txt, content='$reportText'" }

            // 3. Проверка бинарного файла (image)
            val imageEntity = complexRs.getFirst("image") as? HttpEntity<ByteArray>
            val imageBytes = imageEntity?.body
            val isPng = imageBytes?.get(1) == 'P'.code.toByte() // Простейшая проверка сигнатуры PNG

            log.info {
                "Part 'image': size=${imageBytes?.size} bytes, " +
                        "isActuallyBinary=${imageBytes?.contains(0.toByte())}, " +
                        "isPng=$isPng"
            }

            // Итоговый ассерт в логи
            if (statusJson?.contains("all_parts_received") == true && isPng) {
                log.info { "SUCCESS: All multipart response parts correctly parsed and verified!" }
            } else {
                log.error { "FAILURE: Multipart response verification failed!" }
            }
        } else {
            log.error { "FAILURE: Complex multipart response is null" }
        }
        log.info { "Complex multipart test finished. Result exists: ${complexRs != null}" }

        // 5. Тест Huge JSON
        log.info { "--- Test 5: Huge JSON Truncation ---" }
        val hugeRsBytes = extServiceClient.httpClient.sendRq(
            path = "/v1/data/huge",
            method = HttpMethod.GET,
            responseClass = ByteArray::class.java // Читаем байты, Jackson не лезет
        )
        val hugeRs = hugeRsBytes?.let { String(it) } // Превращаем в строку сами
        log.info { "Huge response received, actual size: ${hugeRs?.length} characters" }

        return "All tests executed! Check logs for details."
    }

    @PostMapping("/test-gzip")
    fun checkGzip(): Map<String, Any> {
        log.info { "Starting GZIP integration test (Automatic compression)..." }

        // Вызываем клиент. Netty сам добавит Accept-Encoding (распаковка ответа),
        // а наш GzipRequestInterceptor сожмет тело запроса.
        val response = extServiceClient.testGzipTransfer(MyDataRequest(name = "Payload for auto-compression"))

        return mapOf(
            "status" to if (response?.status == "gzip_success") "OK" else "ERROR",
            "payload" to (response ?: "null")
        )
    }

    @PostMapping("/test-plain-to-gzip")
    fun checkPlainToGzip(): Map<String, Any> {
        log.info { "Starting test: Plain Request -> GZIP Response..." }

        val response = extServiceClient.testPlainToGzip(MyDataRequest(name = "I am plain text"))

        return mapOf(
            "status" to (if (response?.status == "success") "OK" else "ERROR"),
            "received_data" to (response ?: "null"),
        )
    }

    data class MyData(
        var name: String? = null,
        var secondName: String? = null,
        var age: Int = 0,
        var user: User? = null
    )

    companion object {
        private const val PART_RESPONSE_DATA = "responseData"
        private const val PART_RESPONSE_FILE_INFO = "fileInfo"
        private const val PART_RESPONSE_BINARY = "binaryFile"

        private const val INFO_FILENAME = "info.txt"
        private const val BINARY_FILENAME = "image.png"
    }
}
