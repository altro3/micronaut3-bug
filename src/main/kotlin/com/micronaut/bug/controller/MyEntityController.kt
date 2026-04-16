package com.micronaut.bug.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.config.SecurityContext
import com.micronaut.bug.config.User
import com.micronaut.bug.service.BusinessService
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.io.ByteArrayResource
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.util.LinkedMultiValueMap
import org.springframework.util.MultiValueMap
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
class MyEntityController(
    private val businessService: BusinessService,
    private val objectMapper: ObjectMapper
) {

    private val log = KotlinLogging.logger {}

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
