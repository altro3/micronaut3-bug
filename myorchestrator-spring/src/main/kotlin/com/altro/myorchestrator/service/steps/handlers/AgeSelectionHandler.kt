package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpAsyncClient // 🎯 ИСПРАВЛЕНО: Инжектим асинхронный клиент
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class AgeSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpAsyncClient,
) {
    private val log = KotlinLogging.logger {}

    fun processAgeTargeting(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val rawAges = userInput.split(",").map { it.trim() }.filter { it.isNotBlank() }

        if (rawAges.isEmpty()) {
            sink.next("⚠️ Пожалуйста, укажите возрастные ограничения, например: **18+**, **0+** или **Без ограничений**:")
            sink.complete()
            return
        }

        val yadId = session.context.yadCampaignId

        // Если это Яндекс — пушим ID возрастных категорий во внешний service-yad через MCP
        if (session.context.selectedPlatform == Platform.YANDEX_DIRECT && yadId != null) {
            try {
                sink.next("📡 Синхронизирую возрастной таргетинг с service-yad через MCP-инструмент...\n")

                // Вызываем наш новый, красивый однострочный инструмент MCP
                val mcpRequest = CallToolRequest(
                    "bindYandexAges",
                    mapOf("campaignId" to yadId, "ageIds" to rawAges),
                    mapOf()
                )
                mcpClient.callTool(mcpRequest).block()

                log.info { "Успешно выполнен RPC-вызов bindYandexAges для ID: $yadId" }
            } catch (e: Exception) {
                log.error(e) { "Ошибка отправки возрастного таргетинга в service-yad" }
                sink.next("⚠️ Предупреждение: Не удалось автоматически привязать возраст в адаптере Яндекса: ${e.message}\n")
            }
        }

        // Чистая мутация var-свойств нашей модели и фиксация промежуточного шага
        session.currentStep = CampaignCreationStep.AUDIENCE_TARGETING // Переводим на шаг Брифа
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("✅ Возрастные ограничения успешно зафиксированы: ${rawAges.joinToString()}.\n\n")
        sink.next("🤖 **[Шаг 3/5]**: Теперь отправьте мне подробный **текстовый бриф** вашего продукта для ИИ-анализа аудитории:")
        sink.complete()
    }
}
