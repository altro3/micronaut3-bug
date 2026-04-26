package com.micronaut.bug.controller

import com.micronaut.bug.service.integration.internalservice.InternalServiceClient
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class ServiceToServiceController(
    private val internalServiceClient: InternalServiceClient,
) {

    @GetMapping("/service2service/ping")
    fun ping(): Unit =
        internalServiceClient.ping()
}
