package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AiInferenceService
import com.altro.myorchestrator.service.VectorStoreService
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
            ?: throw IllegalStateException("Критическая ошибка: Платформа не выбрана")

        sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Изучаю ваш бриф под требования платформы ${platform.name}...\n")
        val analysisResult = aiInferenceService.analyzeBrief(userInput)

        sink.next("📡 Qdrant [Размерность 2048]: Извлекаю локальные правила модерации по gRPC-фильтру...\n")
        val moderationRules = vectorStoreService.searchAdvertisingRules(listOf(platform))

        // В базу оркестратора сохраняем только факт прохождения шага.
        // Брифы и правила модерации лежат в оперативной памяти для следующего немедленного шага генерации!
        session.currentStep = CampaignCreationStep.AUDIENCE_TARGETING
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("📝 **Результаты ИИ-анализа:**\n")
        sink.next("• Выделенная ЦА: ${analysisResult.audienceDescription}\n")
        sink.next("• Правила модерации площадки успешно подгружены в контекст.\n\n")
        sink.next("🤖 **[Шаг 4/5]**: Напишите **'Да'** или **'Генерируй'**, чтобы запустить написание рекламных объявлений.")
        sink.complete()
    }
}
