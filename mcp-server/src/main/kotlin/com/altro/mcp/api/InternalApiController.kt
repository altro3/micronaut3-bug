package com.altro.mcp.api

import com.altro.mcp.service.integration.extservice.ExtServiceClient
import com.altro.mcp.service.integration.extservice.api.MyDataRequest
import com.altro.mcp.service.integration.extservice.api.MyDataResponse
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class InternalApiController(
    private val extServiceClient: ExtServiceClient,
) {

    private val log = KotlinLogging.logger {}

    @GetMapping("/internal/ping")
    fun process(): MyDataResponse? {
        log.info { "== Service request == Received call from parent" }

        val updateRs = extServiceClient.updateData(MyDataRequest(name = "Test item"))

        return updateRs
    }

    @GetMapping("/internal/error")
    fun error(): MyDataResponse? {
        throw RuntimeException("Произошла какая-то ошибка")
    }
}