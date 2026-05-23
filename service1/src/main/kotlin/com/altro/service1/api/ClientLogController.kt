package com.altro.service1.api

import com.altro.service1.service.integration.extservice.ExtServiceClient
import com.altro.service1.service.integration.extservice.api.MyDataRequest
import com.altro.service1.service.integration.extservice.api.MyDto
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.io.ByteArrayResource
import org.springframework.http.HttpEntity
import org.springframework.http.HttpMethod
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController
import java.io.ByteArrayInputStream
import java.util.zip.GZIPInputStream

@RestController
class ClientLogController(
    private val extServiceClient: ExtServiceClient,
) {

    private val log = KotlinLogging.logger {}

    @GetMapping("/ping")
    fun testPing() {
        // 1. Тест Ping (GET без тела)
        log.info { "--- Test 1: Ping ---" }
        extServiceClient.ping()
    }

    @GetMapping("/json")
    fun testJson() {
        // 2. Тест JSON (POST JSON -> JSON)
        log.info { "--- Test 2: Update Data (JSON) ---" }
        val updateRs = extServiceClient.updateData(MyDataRequest(name = "Test item"))
        log.info { "Update result: $updateRs" }
    }

    @GetMapping("/binary")
    fun testBinary() {
        // 3. Тест Process File (POST Binary -> Binary)
        log.info { "--- Test 3: Process File (Binary) ---" }

        // Создаем реальные бинарные данные, начинающиеся с NULL-байта
        val binaryContent = byteArrayOf(1, 2, 3, 4, 5, 0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte())

        val binaryFile = object : ByteArrayResource(binaryContent) {
            override fun getFilename() = "actual-binary-data.bin"
        }

        val fileRs = extServiceClient.processFile(binaryFile)

        log.info { "Binary request sent. Received response size: ${fileRs?.size ?: 0} bytes" }
    }

    @GetMapping("/multipart")
    fun testMultipart() {
        // 4. Тест Complex Multipart (Path Var + Query + Different Parts)
        log.info { "--- Test 4: Complex Multipart ---" }
        val complexRs = extServiceClient.sendComplexMultipart(
            id = "RQ-777",
            queryTag = "test-tag",
            jsonWithCt = MyDto("key", "val"),
            jsonNoCt = MyDto("k", "v"),
            binaryFile = object : ByteArrayResource(byteArrayOf(1, 2, 3)) {
                override fun getFilename() = "req.bin"
            },
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
    }

    @GetMapping("/huge-json")
    fun testHugeJson() {

        // 5. Тест Huge JSON
        log.info { "--- Test 5: Huge JSON Truncation ---" }
        val hugeRsBytes = extServiceClient.httpClient.sendRq(
            path = "/v1/data/huge",
            method = HttpMethod.GET,
            responseClass = ByteArray::class.java // Читаем байты, Jackson не лезет
        )
        val hugeRs = hugeRsBytes?.let { String(it) } // Превращаем в строку сами
        log.info { "Huge response received, actual size: ${hugeRs?.length} characters" }
    }

    @GetMapping("/gzip")
    fun testGzip(): Map<String, Any> {
        log.info { "Starting GZIP integration test (Automatic compression)..." }

        // Вызываем клиент. Netty сам добавит Accept-Encoding (распаковка ответа),
        // а наш GzipRequestInterceptor сожмет тело запроса.
        val response = extServiceClient.testGzipTransfer(MyDataRequest(name = "Payload for manual compression"))

        return mapOf(
            "status" to "ok",
            "payload" to String(decompress(response!!)),
        )
    }

    @GetMapping("/gzip-auto")
    fun testGzipAuto(): Map<String, Any> {
        log.info { "Starting GZIP-AUTO integration test (Automatic compression)..." }

        // Вызываем клиент. Netty сам добавит Accept-Encoding (распаковка ответа),
        // а наш GzipRequestInterceptor сожмет тело запроса.
        val response = extServiceClient.testAutoGzipTransfer(MyDataRequest(name = "Payload for auto-compression"))

        return mapOf(
            "status" to if (response?.status == "gzip_success") "OK" else "ERROR",
            "payload" to (response ?: "null")
        )
    }

    @GetMapping("/plain-to-gzip")
    fun testPlainToGzip(): Map<String, Any> {
        log.info { "Starting test: Plain Request -> GZIP Response..." }

        val response = extServiceClient.testPlainToGzip(MyDataRequest(name = "I am plain text"))

        return mapOf(
            "status" to (if (response?.status == "success") "OK" else "ERROR"),
            "received_data" to (response ?: "null"),
        )
    }

    @GetMapping("/test-error")
    fun testError(): String {
        log.info { "Starting test: error..." }

        try {
            extServiceClient.testError(MyDataRequest(name = "I am plain text"))
        } catch (e: Exception) {
            //
        }

        return "OK!"
    }

    @GetMapping("/huge-log")
    fun testHugeLog(): Map<String, Any> {
        log.info { "--- Starting Huge Data Protection Test ---" }

        // Генерируем 6МБ данных (больше, чем наш maxPayloadSize)
        val hugePayload = ByteArray(6 * 1024 * 1024) { 0x42.toByte() } // Буква 'B'

        val startTime = System.currentTimeMillis()
        val result = extServiceClient.testHugeData(hugePayload)
        val duration = System.currentTimeMillis() - startTime

        return mapOf(
            "status" to "COMPLETED",
            "requestSize" to hugePayload.size,
            "responseSize" to (result?.size ?: 0),
            "durationMs" to duration
        )
    }

    private fun decompress(bytes: ByteArray): ByteArray {
        return try {
            GZIPInputStream(ByteArrayInputStream(bytes)).use { it.readBytes() }
        } catch (e: Exception) {
            log.warn { "Failed to decompress GZIP body, logging raw data. Error: ${e.message}" }
            bytes
        }
    }

}
