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
import io.github.oshai.kotlinlogging.KotlinLogging
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
    private val mcpPublishingHandler: McpPublishingHandler,
    private val dynamicAiOrchestrator: DynamicAiOrchestrator
) {

    private val log = KotlinLogging.logger {}

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
                    if (userInput.contains("умный", ignoreCase = true) || userInput.contains("ии", ignoreCase = true)) {
                        val newSession = CampaignSession(currentStep = CampaignCreationStep.AI_DYNAMIC_COLLECTING)
                        dynamicAiOrchestrator.orchestrateDynamicStream(newSession, userInput)
                            .subscribe(sink::next, sink::error, sink::complete)
                    } else {
                        platformSelectionHandler.initializeSession(firstMessage, userInput, sink) // Старый путь
                    }
                } else {
                    if (session.currentStep == CampaignCreationStep.AI_DYNAMIC_COLLECTING) {
                        dynamicAiOrchestrator.orchestrateDynamicStream(session, userInput).subscribe(sink::next, sink::error, sink::complete)
                    } else {
                        executeSessionStep(session, userInput, sink)
                    }
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
            CampaignCreationStep.REGION_SELECTION -> ageSelectionHandler.processAgeTargeting(session, userInput, sink)
            CampaignCreationStep.AUDIENCE_TARGETING -> audienceTargetingHandler.processAudience(session, userInput, sink)
            CampaignCreationStep.CREATIVE_GENERATION -> creativeGenerationHandler.processCreatives(session, userInput, sink, ::isUserApproved)
            CampaignCreationStep.PUBLISHING_AND_PAYMENT -> mcpPublishingHandler.processPublish(session, userInput, sink, ::isUserApproved)
            CampaignCreationStep.COMPLETED -> mcpPublishingHandler.handleCompleted(sink)

            // НОВАЯ СОВРЕМЕННАЯ ВЕТКА: Если сессия уже идет по ИИ-пути
            CampaignCreationStep.AI_DYNAMIC_COLLECTING,
            CampaignCreationStep.AI_DYNAMIC_CONFIRMATION -> {
                log.info { "Перенаправляю запрос в динамический ИИ-оркестратор для сессии: ${session.campaignId}" }

                // Вызываем новый оркестратор, который возвращает Flux<String>
                dynamicAiOrchestrator.orchestrateDynamicStream(session, userInput)
                    .subscribe(
                        { chunk -> sink.next(chunk) },     // Передаем каждый ИИ-токен в текущий веб-сокет/SSE
                        { error -> sink.error(error) },    // Пробрасываем ошибку в пайплайн чата
                        { sink.complete() }                // Закрываем стрим, когда ИИ закончил говорить
                    )
            }
        }
    }



    private fun isUserApproved(input: String): Boolean {
        val clean = input.lowercase().trim()
        return clean in listOf("да", "ок", "публикуй", "подтверждаю", "yes", "ok", "генерируй")
    }
}
