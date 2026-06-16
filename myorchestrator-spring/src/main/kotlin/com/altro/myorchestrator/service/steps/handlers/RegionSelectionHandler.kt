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
        // Очищаем ввод от лишних пробелов и разбиваем по запятым
        val rawRegions = userInput.split(",").map { it.trim() }.filter { it.isNotBlank() }

        if (rawRegions.isEmpty()) {
            sink.next("⚠️ Пожалуйста, укажите хотя бы один регион или город через запятую (например: *Москва, Нижний Новгород*):")
            sink.complete()
            return
        }

        val yadId = session.context.yadCampaignId

        // Если кампания создается под Яндекс — пушим человеческие строки регионов через MCP.
        // Адаптер service-yad сам внутри себя сопоставит их по ID из Caffeine кэша.
        if (session.context.selectedPlatform == Platform.YANDEX_DIRECT && yadId != null) {
            try {
                sink.next("📡 Передаю гео-таргетинг в service-yad через MCP-гейтвей...\n")

                val mcpRequest = CallToolRequest(
                    "bindYandexRegions",
                    mapOf("campaignId" to yadId, "regionIds" to rawRegions),
                    mapOf()
                )
                mcpClient.callTool(mcpRequest)
                log.info { "Успешно выполнен RPC-вызов bindYandexRegions для ID: $yadId" }
            } catch (e: Exception) {
                log.error(e) { "Ошибка синхронизации гео-таргетинга с service-yad" }
                sink.next("⚠️ Предупреждение: Не удалось автоматически привязать регионы в адаптере Яндекса: ${e.message}\n")
            }
        }

        // Чистая мутация var-параметров стейта без оверхеда пересоздания объектов
        session.currentStep = CampaignCreationStep.REGION_SELECTION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("✅ Регионы таргетинга зафиксированы: ${rawRegions.joinToString()}.\n\n")
        sink.next("🤖 **[Шаг 3/5]**: Теперь отправьте мне подробный **текстовый бриф** вашего продукта (описание, цели, особенности ЦА). ИИ-анализатор изучит его:")
        sink.complete()
    }
}
