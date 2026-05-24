package com.altro.common.util.api

import com.altro.common.util.api.dto.ErrorRs
import com.altro.common.trace.NanoTracer
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.server.ResponseStatusException

open class DefExceptionHandler(
    private val tracer: NanoTracer? = null
) {
    private val log = KotlinLogging.logger {}

    /**
     * Системные ошибки (500)
     */
    @ExceptionHandler(Exception::class)
    open fun handleAll(ex: Exception): ResponseEntity<ErrorRs> {
        val span = tracer?.currentSpan()
        span?.error = ex

        log.error(ex) { "System error: ${ex.message}" }

        return ResponseEntity
            .status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(
                ErrorRs(
                    rqId = span?.traceId,
                    code = "INTERNAL_SERVER_ERROR",
                    message = ex.message
                )
            )
    }

    /**
     * Ошибки валидации или бизнес-логики (400)
     */
    @ExceptionHandler(IllegalArgumentException::class)
    open fun handleBadRequest(ex: IllegalArgumentException): ResponseEntity<ErrorRs> {
        val span = tracer?.currentSpan()
        // Даже для 400-х ошибок лучше писать exception в трейс для отладки
        span?.error = ex

        return ResponseEntity
            .status(HttpStatus.BAD_REQUEST)
            .body(
                ErrorRs(
                    rqId = span?.traceId,
                    code = "BAD_REQUEST",
                    message = ex.message
                )
            )
    }

    /**
     * Обработка ошибок с заданным статусом (например, через ResponseStatusException)
     */
    @ExceptionHandler(ResponseStatusException::class)
    open fun handleResponseStatus(ex: ResponseStatusException): ResponseEntity<ErrorRs> {
        val span = tracer?.currentSpan()
        span?.error = ex

        log.error(ex) { "Error" }

        return ResponseEntity
            .status(ex.statusCode)
            .body(
                ErrorRs(
                    rqId = span?.traceId,
                    code = "HTTP_${ex.statusCode.value()}",
                    message = ex.reason ?: ex.message
                )
            )
    }
}