package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.domain.CampaignCreationStep
import com.altro.myorchestrator.domain.CampaignSession
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AiInferenceService
import com.altro.myorchestrator.service.VectorStoreService
import com.altro.myorchestrator.service.integration.AiInferenceService
import com.altro.myorchestrator.service.integration.VectorStoreService
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class AudienceTargetingHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val aiInferenceService: AiInferenceService,
    private val vectorStoreService: VectorStoreService
) {

    fun processAudience(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val platform = session.context.selectedPlatform 
            ?: throw IllegalStateException("Критическая ошибка: платформа не выбрана")

        sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Изучаю бриф под требования платформы ${platform.name}...\n")
        val analysisResult = aiInferenceService.analyzeBrief(userInput)
        
        sink.next("📡 Qdrant [Размерность 2048]: Извлекаю правила модерации по gRPC-фильтру...\n")
        val moderationRules = vectorStoreService.searchAdvertisingRules(listOf(platform))

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.AUDIENCE_TARGETING,
                context = session.context.copy(
                    targetAudienceDescription = analysisResult.audienceDescription,
                    analysisResult = analysisResult,
                    moderationRules = moderationRules
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("📝 **Результаты анализа:**\n")
        sink.next("• ЦА: ${analysisResult.audienceDescription}\n")
        sink.next("• Лимиты и правила модерации успешно подгружены из локальной базы знаний.\n\n")
        sink.next("🤖 **[Шаг 4/5]**: Напишите **'Да'** или **'Генерируй'**, чтобы ИИ написал продающий креатив.")
        sink.complete()
    }
}
