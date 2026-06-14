package com.altro.myorchestrator.api

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.api.dto.chat.ChatRs
import com.altro.myorchestrator.service.ChatOrchestrator
import jakarta.validation.Valid
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class ApiController(
    private val orchestrator: ChatOrchestrator
) {

    @PostMapping("/api/chat/completions")
    fun chatCompletions(@RequestBody @Valid rq: ChatRq): ChatRs =
        orchestrator.orchestrateChat(rq)
}
