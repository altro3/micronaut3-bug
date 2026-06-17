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

        // 1. Сохраняем текст брифа в историю (первое сообщение юзера на этом шаге)
        // Чтобы модель на следующем шаге генерации креативов видела, про что писать рекламу
        val updatedLogs = session.context.executionLogs.toMutableList().apply {
            add(userInput)
        }

        // Вызов локального инференса Qwen-35B для анализа ЦА
        val analysisResult = aiInferenceService.analyzeBrief(userInput)

        sink.next("📡 Qdrant [Размерность 2048]: Извлекаю актуальные правила модерации из базы знаний...\n")

        // 2. 🎯 ИСПРАВЛЕНО: Передаем платформу и живой бриф (userInput) для поиска категории бизнеса (ALL)
        val extractedRules = vectorStoreService.searchAdvertisingRules(
            platforms = listOf(platform),
            userBriefText = userInput
        )
        log.info { "Для кампании ${session.campaignId} из Qdrant извлечены правила модерации." }

        // 3. Добавляем ответ ИИ и извлеченные правила в историю логов сессии,
        // чтобы на Шаге 4 CreativeGeneratorProcessor смог их прочитать из базы.
        updatedLogs.add(analysisResult.audienceDescription)
        // Пакуем скрытый системный контекст правил в логи, чтобы передать его на следующий шаг
        updatedLogs.add("SYSTEM_RULES_CONTEXT: $extractedRules")
        session.context.executionLogs = updatedLogs

        // 4. 🎯 ИСПРАВЛЕНО: Переводим сессию на ШАГ КРЕАТИВОВ, убирая вечный цикл
        session.currentStep = CampaignCreationStep.CREATIVE_GENERATION
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
