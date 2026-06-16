package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform // Наш Enum площадок
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import java.time.Instant

@Component
class PlatformSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpSyncClient,
    private val jsonMapper: JsonMapper
) {

    private val log = KotlinLogging.logger {}

    fun sendWelcomeMessage(sink: FluxSink<String>) {
        sink.next("👋 Привет! Я локальный ИИ-ассистент автоматизации маркетинга AdBroker.\n")
        sink.next("Давайте пошагово настроим вашу рекламную кампанию.\n\n")
        sink.next("🤖 **[Шаг 1/5]**: Выберите целевую рекламную платформу. Напишите **VK_ADS** или **YANDEX_DIRECT**:")
        sink.complete()
    }

    fun initializeSession(firstMessage: String, userInput: String, sink: FluxSink<String>) {
        val platform = parsePlatform(userInput)

        if (platform == null) {
            sink.next("⚠️ Пожалуйста, напишите корректно одну из доступных платформ: **VK_ADS** или **YANDEX_DIRECT**.")
            sink.complete()
            return
        }

        sink.next("⚙️ Инициирую черновик и создаю сессию в serviceYad через MCP-инструмент...\n")

        var yadId: Long = 0L

        // Сравниваем строго по типам Enum, а не по строкам
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                val mcpRequest = CallToolRequest(
                    "initYandexDraft",
                    mapOf("name" to "Кампания_${System.currentTimeMillis() / 1000}"),
                    mapOf()
                )
                val mcpResponse = mcpClient.callTool(mcpRequest)

                val rootNode = jsonMapper.readTree(mcpResponse.content.toString())
                yadId = rootNode.get("id")?.asLong() ?: throw IllegalStateException("Адаптер Яндекса вернул пустой ID")

                log.info { "Успешно вызван MCP initYandexDraft. Получен сквозной ID кампании: $yadId" }
            } catch (e: Exception) {
                log.error(e) { "Сбой инициализации кампании на стороне serviceYad" }
                sink.next("❌ Критическая ошибка: Не удалось связаться с serviceYad через MCP. Черновик не создан. Причина: ${e.message}")
                sink.complete()
                return
            }
        } else {
            // Фолбэк для VK Ads, пока не поднят отдельный микросервис
            yadId = System.currentTimeMillis()
        }

        val session = CampaignSession(
            campaignId = yadId, // Сквозной ID становится первичным ключом
            currentStep = CampaignCreationStep.PLATFORM_SELECTION,
            updatedAt = Instant.now()
        )

        // Накатываем строго типизированный Enum
        session.context.selectedPlatform = platform
        session.context.yadCampaignId = yadId
        session.context.executionLogs = listOf(firstMessage)

        val saved = sessionRepository.save(session)

        sink.next("✅ Рекламная платформа ${saved.context.selectedPlatform?.name} успешно зафиксирована.\n")
        sink.next("🔹 ID вашей сессии в системе: **${saved.campaignId}**\n\n")
        sink.next("🤖 **[Шаг 2/5]**: Теперь введите **регионы таргетинга** через запятую (например: *Москва, Санкт-Петербург*):")
        sink.complete()
    }

    private fun parsePlatform(input: String): Platform? {
        val cleanInput = input.uppercase().trim()
        return when {
            cleanInput.contains("VK") -> Platform.VK_ADS
            cleanInput.contains("YANDEX") || cleanInput.contains("ЯНДЕКС") -> Platform.YANDEX_DIRECT
            else -> null
        }
    }
}
