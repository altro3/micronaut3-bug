package com.micronaut.bug.controller

import com.micronaut.bug.config.SecurityContext
import com.micronaut.bug.config.User
import com.micronaut.bug.service.BusinessService
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
class MyEntityController(
    private val businessService: BusinessService
) {

    private val log = KotlinLogging.logger {}

    @PostMapping("/testMultipart/{pathVar}")
    fun testMultipart(
        @PathVariable pathVar: String,
        @RequestParam queryVar: String,
        @RequestBody(required = false) data: MyData?
    ): MyData {
        log.info { "body: ${data ?: "null"}" }
        log.info { "pathVar: $pathVar" }
        log.info { "queryVar: $queryVar" }

        val currentUser = SecurityContext.getUser()
        val processedUser = businessService.processOrder(currentUser)

        // Используем copy() или возвращаем новый объект, соблюдая иммутабельность,
        // либо работаем с mutable полем, если это необходимо.
        return (data ?: MyData()).apply {
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
