package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.steps.handlers.AgeSelectionHandler
import com.altro.myorchestrator.service.steps.handlers.AudienceTargetingHandler
import com.altro.myorchestrator.service.steps.handlers.CreativeGenerationHandler
import com.altro.myorchestrator.service.steps.handlers.McpPublishingHandler
import com.altro.myorchestrator.service.steps.handlers.PlatformSelectionHandler
import com.altro.myorchestrator.service.steps.handlers.RegionSelectionHandler
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.FluxSink
import reactor.core.scheduler.Schedulers

@Service
class ChatOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val platformSelectionHandler: PlatformSelectionHandler,
    private val regionSelectionHandler: RegionSelectionHandler,
    private val ageSelectionHandler: AgeSelectionHandler,
    private val audienceTargetingHandler: AudienceTargetingHandler,
    private val creativeGenerationHandler: CreativeGenerationHandler,
    private val mcpPublishingHandler: McpPublishingHandler
) {

    fun orchestrateChatStream(rq: ChatRq): Flux<String> =
        Flux.create { sink ->
            try {
                if (rq.messages.isEmpty()) {
                    platformSelectionHandler.sendWelcomeMessage(sink)
                    return@create
                }

                val firstMessage = rq.messages.first().content.trim()
                val userInput = rq.messages.last().content.trim()

                val session = sessionRepository.findByFirstMessage(firstMessage)

                if (session == null) {
                    platformSelectionHandler.initializeSession(firstMessage, userInput, sink)
                } else {
                    executeSessionStep(session, userInput, sink)
                }

            } catch (e: Exception) {
                sink.next("❌ Критический сбой пайплайна чата: ${e.message}")
                sink.error(e)
            }
        }.subscribeOn(Schedulers.boundedElastic())

    private fun executeSessionStep(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        when (session.currentStep) {
            CampaignCreationStep.DRAFT -> platformSelectionHandler.initializeSession(session.context.executionLogs.first(), userInput, sink)
            CampaignCreationStep.PLATFORM_SELECTION -> regionSelectionHandler.processRegions(session, userInput, sink)

            // СВЯЗУЮЩЕЕ ЗВЕНО: Теперь шаг регионов перенаправляет поток на обработку возраста
            CampaignCreationStep.REGION_SELECTION -> ageSelectionHandler.processAgeTargeting(session, userInput, sink)

            // После возраста переходим к анализу брифа
            CampaignCreationStep.AUDIENCE_TARGETING -> audienceTargetingHandler.processAudience(session, userInput, sink)

            CampaignCreationStep.CREATIVE_GENERATION -> creativeGenerationHandler.processCreatives(session, userInput, sink, ::isUserApproved)
            CampaignCreationStep.PUBLISHING_AND_PAYMENT -> mcpPublishingHandler.processPublish(session, userInput, sink, ::isUserApproved)
            CampaignCreationStep.COMPLETED -> mcpPublishingHandler.handleCompleted(sink)
        }
    }

    private fun isUserApproved(input: String): Boolean {
        val clean = input.lowercase().trim()
        return clean in listOf("да", "ок", "публикуй", "подтверждаю", "yes", "ok", "генерируй")
    }
}
