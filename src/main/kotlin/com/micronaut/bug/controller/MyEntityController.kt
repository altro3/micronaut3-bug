package com.micronaut.bug.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.config.SecurityContext
import com.micronaut.bug.config.User
import com.micronaut.bug.service.BusinessService
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
class MyEntityController(
    private val businessService: BusinessService,
    private val objectMapper: ObjectMapper,
) {

    private val log = KotlinLogging.logger {}

    @PostMapping("/testMultipart/{pathVar}", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun testMultipart(
        @PathVariable pathVar: String,
        @RequestParam queryVar: String,
//        @RequestBody(required = false) dataParsed: MyData?,
        @RequestPart(required = false) data: String?,
        @RequestPart(required = false) file: MultipartFile?,
    ): MyData {
        val dataParsed = data?.let { objectMapper.readValue(data, MyData::class.java) }
        log.info { "body: ${data ?: "null"}" }
        log.info { "dataParsed: $dataParsed" }
        log.info { "pathVar: $pathVar" }
        log.info { "queryVar: $queryVar" }

        val currentUser = SecurityContext.getUser()
        val processedUser = businessService.processOrder(currentUser)

        // Используем copy() или возвращаем новый объект, соблюдая иммутабельность,
        // либо работаем с mutable полем, если это необходимо.
        return (dataParsed ?: MyData()).apply {
            user = processedUser
        }
    }

    data class MyData(
        var name: String? = null,
        var secondName: String? = null,
        var age: Int = 0,
        var user: User? = null
    )
}
