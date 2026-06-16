package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CreativeGeneratorProcessor
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class CreativeGenerationHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val creativeGeneratorProcessor: CreativeGeneratorProcessor,
) {

    fun processCreatives(session: CampaignSession, userInput: String, sink: FluxSink<String>, isApproved: (String) -> Boolean) {
        if (!isApproved(userInput)) {
            sink.next("⚠️ Напишите **'Да'** или **'Генерируй'**, чтобы запустить генерацию объявления моделью Qwen-35B.")
            sink.complete()
            return
        }

        sink.next("⚡ Запускаю Qwen-35B (n_predict=400). Пишем рекламный креатив...\n")
        val analysis = session.context.analysisResult ?: throw IllegalStateException("Анализ отсутствует")
        val rules = session.context.moderationRules ?: ""

        val creatives = creativeGeneratorProcessor.generate(analysis, rules)

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.CREATIVE_GENERATION,
                context = session.context.copy(generatedCreatives = creatives),
                updatedAt = Instant.now()
            )
        )

        val platform = session.context.selectedPlatform!!
        val creative = creatives[platform]

        sink.next("✨ **Рекламный креатив сформирован:**\n")
        if (creative != null) {
            sink.next("\n📢 **Площадка: ${platform.name}**\n🔹 Заголовок: ${creative.title}\n🔸 Текст: ${creative.bodyText}\n")
        } else {
            sink.next("⚠️ Модель не сгенерировала текст под выбранную платформу.")
        }

        sink.next("\n🤖 **[Шаг 5/5]**: Публикуем это объявление через Stateless MCP-сервер (8090)? Напишите **'Да'** или **'Публикуй'**.")
        sink.complete()
    }
}
