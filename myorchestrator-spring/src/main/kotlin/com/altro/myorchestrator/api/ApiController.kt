package com.altro.myorchestrator.api

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.api.dto.chat.ChatRs
import com.altro.myorchestrator.service.ChatOrchestrator
import jakarta.validation.Valid
import org.springframework.http.MediaType
import org.springframework.http.MediaType.TEXT_EVENT_STREAM_VALUE
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter
import reactor.core.publisher.Flux

@RestController
class ApiController(
    private val orchestrator: ChatOrchestrator
) {

    @PostMapping("/chat/completions")
    fun chatCompletions(@RequestBody @Valid rq: ChatRq): ChatRs =
        orchestrator.orchestrateChat(rq)

    @PostMapping("/chat/completions/stream", produces = [TEXT_EVENT_STREAM_VALUE])
    fun chatCompletionsStream(@RequestBody @Valid rq: ChatRq): Flux<String> =
        orchestrator.orchestrateChatStream(rq)
}
