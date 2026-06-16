package com.altro.service1.api

import com.altro.service1.repository.UserRepository
import com.altro.service1.service.BusinessService
import com.altro.service1.service.integration.extservice.ExtServiceClient
import com.altro.service1.service.integration.extservice.api.MyDataRequest
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.data.repository.findByIdOrNull
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class TracingController(
    private val extServiceClient: ExtServiceClient,
    private val businessService: BusinessService,
    private val jdbcTemplate: JdbcTemplate,
    private val userRepository: UserRepository,
) {

    private val log = KotlinLogging.logger {}

    @GetMapping("/trace/ping")
    fun ping(): String {
        jdbcTemplate.execute("SELECT 1")
        return "pong"
    }

    @GetMapping("/trace/error")
    fun triggerError(): String {

        val updateRs = extServiceClient.updateData(MyDataRequest(name = "Test item"))

        userRepository.findByIdOrNull(10)

        try {
            businessService.someFunc()
        } catch (e: Exception) {
            log.error { "Oops! Something went wrong in our business logic" }
            throw Exception("Большое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\n", e)
        }
        return "OK"
    }
}
