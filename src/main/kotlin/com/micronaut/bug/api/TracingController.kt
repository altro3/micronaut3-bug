package com.micronaut.bug.api

import com.micronaut.bug.repository.UserRepository
import com.micronaut.bug.service.BusinessService
import com.micronaut.bug.service.integration.extservice.ExtServiceClient
import com.micronaut.bug.service.integration.extservice.api.MyDataRequest
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
            // Выбрасываем исключение, чтобы Spring вернул 500 статус,
            // а наш ServerLoggingFilter перехватил это и отправил Error-статус в Tempo.
            throw Exception("Большое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\nБольшое сообщение об ошибке на несколько строк\n", e)
        }
        return "OK"
    }
}
