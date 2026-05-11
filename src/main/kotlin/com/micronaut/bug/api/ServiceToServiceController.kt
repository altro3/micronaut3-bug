package com.micronaut.bug.api

import com.micronaut.bug.service.integration.extservice.ExtServiceClient2
import com.micronaut.bug.service.integration.internalservice.InternalServiceClient
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class ServiceToServiceController(
    private val internalServiceClient: InternalServiceClient,
    private val extServiceClient: ExtServiceClient2,
) {

    @GetMapping("/service2service/ping")
    fun ping() {
        var ex: Exception? = null
        try {
            internalServiceClient.ping()
        } catch (e: Exception) {
            ex = e
        }
        try {
            extServiceClient.ping()
        } catch (e: Exception) {
            ex = e
        }
        if (ex != null) {
            throw ex
        }
    }

    @GetMapping("/service2service/error")
    fun error() {
        internalServiceClient.error()
        extServiceClient.ping()
    }
}
