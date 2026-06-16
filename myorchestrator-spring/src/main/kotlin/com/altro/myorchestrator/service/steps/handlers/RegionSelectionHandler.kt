package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class RegionSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpSyncClient
) {
    private val log = KotlinLogging.logger {}

    fun processRegions(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val regions = userInput.split(",").map { it.trim() }
        val yadId = session.context.yadCampaignId

        // Если это Яндекс — пушим регионы по сети во внешний service-yad
        if (session.context.selectedPlatform == Platform.YANDEX_DIRECT && yadId != null) {
            try {
                sink.next("📡 Синхронизирую регионы с serviceYad через MCP-инструмент...\n")
                val mcpRequest = CallToolRequest(
                    "bindYandexRegions",
                    mapOf("campaignId" to yadId, "regionIds" to regions),
                    mapOf()
                )
                mcpClient.callTool(mcpRequest)
                log.info { "Успешно вызван MCP bindYandexRegions для ID: $yadId" }
            } catch (e: Exception) {
                log.error(e) { "Сбой отправки регионов в serviceYad" }
                sink.next("⚠️ Ошибка: Не удалось передать регионы в serviceYad: ${e.message}\n")
            }
        }

        // Мутируем var свойства напрямую и сохраняем легкий стейт
        session.currentStep = CampaignCreationStep.REGION_SELECTION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("✅ Регионы таргетинга успешно зафиксированы.\n\n")
        sink.next("🤖 **[Шаг 3/5]**: Теперь отправьте подробный **текстовый бриф** вашего продукта для ИИ-анализа аудитории:")
        sink.complete()
    }
}
