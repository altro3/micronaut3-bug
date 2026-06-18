package com.altro.myorchestrator.api

import com.altro.myorchestrator.api.dto.ChatRq
import com.altro.myorchestrator.api.dto.InitSessionRs
import com.altro.myorchestrator.service.ChatOrchestrator
import jakarta.validation.Valid
import org.springframework.http.MediaType.TEXT_EVENT_STREAM_VALUE
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono

@RestController
class ApiController(
    private val orchestrator: ChatOrchestrator
) {

    @PostMapping("/chat/session/init")
    fun initSession(): Mono<InitSessionRs> =
        orchestrator.createNewSession()

    @PostMapping("/chat/completions/stream", produces = [TEXT_EVENT_STREAM_VALUE])
    fun chatCompletionsStream(@RequestBody @Valid rq: ChatRq): Flux<String> =
        orchestrator.orchestrateChatStream(rq)
}
