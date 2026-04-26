package com.micronaut.bug.controller

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class InternalApiController {

    private val log = KotlinLogging.logger {}

    @GetMapping("/internal/ping")
    fun process(): Map<String, String> {
        // Здесь твоя common-либа должна автоматически подхватить rqId из MDC
        log.info { "== Service request == Received call from parent" }

        return mapOf(
            "status" to "OK",
            "service" to "Thi is internal service"
        )
    }
}