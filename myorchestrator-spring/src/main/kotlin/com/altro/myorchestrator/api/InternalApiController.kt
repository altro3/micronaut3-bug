package com.altro.myorchestrator.api

import com.altro.myorchestrator.api.dto.BriefRequest
import com.altro.myorchestrator.service.MarketingOrchestratorEngineOld
import jakarta.validation.Valid
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter
import java.util.concurrent.Executors

//@RequestMapping("/api/v1")
//@RestController
class InternalApiController(
    private val engine: MarketingOrchestratorEngineOld
) {

    @PostMapping("/orchestrate")
    fun orchestrate(
        @RequestBody @Valid request: BriefRequest
    ): SseEmitter {
        val emitter = SseEmitter(600000L)
        val loomDispatcher = Executors.newVirtualThreadPerTaskExecutor().asCoroutineDispatcher()

        CoroutineScope(loomDispatcher).launch {
            try {
                engine.executeOrchestration(request.text) { currentState ->
                    emitter.send(
                        SseEmitter.event()
                            .name("status")
                            .data(currentState)
                    )
                }
                emitter.complete()
            } catch (ex: Exception) {
                emitter.completeWithError(ex)
            } finally {
                loomDispatcher.close()
            }
        }

        return emitter
    }
}
