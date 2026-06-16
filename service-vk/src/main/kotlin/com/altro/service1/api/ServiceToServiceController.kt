package com.altro.service1.api

import com.altro.service1.repository.UserRepository
import com.altro.service1.service.integration.extservice.ExtServiceClient2
import com.altro.service1.service.integration.service2.Service2Client
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class ServiceToServiceController(
    private val service2Client: Service2Client,
    private val extServiceClient: ExtServiceClient2,
    private val userRepository: UserRepository,
) {

    @GetMapping("/service2service/ping")
    fun ping() {
        var ex: Exception? = null
        try {
            service2Client.ping()
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
        userRepository.findAll()
    }

    @GetMapping("/service2service/error")
    fun error() {
        service2Client.error()
        extServiceClient.ping()
    }
}
