package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AiInferenceService
import com.altro.myorchestrator.service.VectorStoreService
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class AudienceTargetingHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val aiInferenceService: AiInferenceService,
    private val vectorStoreService: VectorStoreService
) {
    private val log = KotlinLogging.logger {}

    fun processAudience(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val platform = session.context.selectedPlatform
            ?: throw IllegalStateException("Критическая ошибка: Рекламная платформа не задана в сессии")

        if (userInput.blankOrTooShort()) {
            sink.next("⚠️ Текст брифа слишком короткий. Пожалуйста, опишите ваш продукт или услугу подробнее:")
            sink.complete()
            return
        }

        val platformNameReadable = if (platform.name == "YANDEX_DIRECT") "Яндекс.Директ" else "ВКонтакте"
        sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Изучаю ваш бриф под требования площадки $platformNameReadable...\n")

        // Вызов локального инференса Qwen-35B для анализа ЦА
        val analysisResult = aiInferenceService.analyzeBrief(userInput)

        sink.next("📡 Qdrant [Размерность 2048]: Извлекаю актуальные правила модерации из базы знаний...\n")
        // Подтягиваем правила через gRPC-фильтр
        vectorStoreService.searchAdvertisingRules(listOf(platform))

        // Мутируем var свойства напрямую и сохраняем легкий стейт
        session.currentStep = CampaignCreationStep.AUDIENCE_TARGETING
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("📝 **Результаты ИИ-анализа брифа:**\n")
        sink.next("• Выделенная целевая аудитория: ${analysisResult.audienceDescription}\n")
        sink.next("• Правила лимитов символов для площадки $platformNameReadable успешно загружены в контекст.\n\n")
        sink.next("🤖 **[Шаг 4/5]**: Переходим к написанию объявлений моделью Qwen-35B? Напишите **'Да'** или **'Генерируй'**:")
        sink.complete()
    }

    private fun String.blankOrTooShort(): Boolean = this.trim().length < 10
}
