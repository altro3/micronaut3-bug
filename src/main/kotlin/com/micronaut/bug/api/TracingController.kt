package com.micronaut.bug.api

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class TracingController {

    private val log = KotlinLogging.logger {}

    @GetMapping("/trace/ping")
    fun ping() = "pong"

    @GetMapping("/trace/error")
    fun triggerError(): String {
        log.error { "Oops! Something went wrong in our business logic" }

        // Выбрасываем исключение, чтобы Spring вернул 500 статус,
        // а наш ServerLoggingFilter перехватил это и отправил Error-статус в Tempo.
        throw RuntimeException("Simulated service failure for tracing test")
    }
}
