package com.micronaut.bug.api

import com.micronaut.bug.api.dto.ErrorRs
import com.micronaut.bug.trace.NanoTracer
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice
import org.springframework.web.server.ResponseStatusException

@RestControllerAdvice
class ExceptionHandler(
    private val tracer: NanoTracer
) {
    private val log = KotlinLogging.logger {}

    /**
     * Системные ошибки (500)
     */
    @ExceptionHandler(Exception::class)
    fun handleAll(ex: Exception): ResponseEntity<ErrorRs> {
        val ctx = tracer.currentContext()
        ctx?.error = ex // Фиксируем для Tempo

        log.error(ex) { "System error: ${ex.message}" }

        return ResponseEntity
            .status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(
                ErrorRs(
                    rqId = ctx?.traceId,
                    code = "INTERNAL_SERVER_ERROR",
                    message = ex.message
                )
            )
    }

    /**
     * Ошибки валидации или бизнес-логики (400)
     */
    @ExceptionHandler(IllegalArgumentException::class)
    fun handleBadRequest(ex: IllegalArgumentException): ResponseEntity<ErrorRs> {
        val ctx = tracer.currentContext()
        // Даже для 400-х ошибок лучше писать exception в трейс для отладки
        ctx?.error = ex

        return ResponseEntity
            .status(HttpStatus.BAD_REQUEST)
            .body(
                ErrorRs(
                    rqId = ctx?.traceId,
                    code = "BAD_REQUEST",
                    message = ex.message
                )
            )
    }

    /**
     * Обработка ошибок с заданным статусом (например, через ResponseStatusException)
     */
    @ExceptionHandler(ResponseStatusException::class)
    fun handleResponseStatus(ex: ResponseStatusException): ResponseEntity<ErrorRs> {
        val ctx = tracer.currentContext()
        ctx?.error = ex

        log.error(ex) { "Error" }

        return ResponseEntity
            .status(ex.statusCode)
            .body(
                ErrorRs(
                    rqId = ctx?.traceId,
                    code = "HTTP_${ex.statusCode.value()}",
                    message = ex.reason ?: ex.message
                )
            )
    }
}
