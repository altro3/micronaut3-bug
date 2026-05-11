package com.micronaut.bug.api

import com.micronaut.bug.service.BusinessService
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class TracingController(
    private val businessService: BusinessService,
) {

    private val log = KotlinLogging.logger {}

    @GetMapping("/trace/ping")
    fun ping() = "pong"

    @GetMapping("/trace/error")
    fun triggerError(): String {

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
