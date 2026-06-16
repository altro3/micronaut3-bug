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
        sink.next("🤖 **[Шаг 1/5]**: На какой площадке запускаем рекламу? Напишите, например: **Яндекс** или **ВК**:")
        sink.complete()
    }

    fun initializeSession(firstMessage: String, userInput: String, sink: FluxSink<String>) {
        val platform = parsePlatform(userInput)

        if (platform == null) {
            sink.next("⚠️ Не совсем понял площадку. Пожалуйста, напишите понятнее, например: **Яндекс**, **Яд**, **ВК** или **ВКонтакте**.")
            sink.complete()
            return
        }

        sink.next("⚙️ Инициирую сессию в service-yad через MCP-гейтвей...\n")
        var yadId = System.currentTimeMillis() // Фолбэк

        if (platform == Platform.YANDEX_DIRECT) {
            try {
                // Создаем драфт в адаптере Яндекса через МСР
                val mcpRequest = CallToolRequest("initYandexDraft", mapOf("name" to "Кампания_${System.currentTimeMillis() / 1000}"), mapOf())
                val mcpResponse = mcpClient.callTool(mcpRequest)

                val rootNode = jsonMapper.readTree(mcpResponse.content.toString())
                yadId = rootNode.get("id")?.asLong() ?: throw IllegalStateException("ID утерян в ответе service-yad")
                log.info { "Успешно вызван MCP initYandexDraft. Локальный ID в адаптере: $yadId" }
            } catch (e: Exception) {
                log.error(e) { "Ошибка МСР-транспорта на Шаге 1" }
                sink.next("❌ Ошибка МСР-транспорта: ${e.message}\n")
                sink.complete()
                return
            }
        }

        // Создаем и мутируем var-модель сессии оркестратора чата
        val session = CampaignSession(
            campaignId = yadId, // Сквозной ID становится первичным ключом сессии оркестратора
            currentStep = CampaignCreationStep.PLATFORM_SELECTION,
            updatedAt = Instant.now()
        ).apply {
            context.selectedPlatform = platform
            context.yadCampaignId = yadId
            context.executionLogs = listOf(firstMessage)
        }

        sessionRepository.save(session)

        val platformReadableName = if (platform == Platform.YANDEX_DIRECT) "Яндекс.Директ" else "ВКонтакте"
        sink.next("✅ Отлично! Рекламная платформа **$platformReadableName** зафиксирована в системе.\n")

        // Короткая, лаконичная реплика.React-фронтенд сам подрисует комбобокс городов на основе currentStep
        sink.next("🤖 **[Шаг 2/5]**: Пожалуйста, выберите **регионы таргетинга** из появившегося списка в интерфейсе формы:")
        sink.complete()
    }

    private fun parsePlatform(input: String): Platform? {
        val clean = input.lowercase().trim()
        return when {
            clean.contains("яндекс") || clean.contains("yad") || clean.contains("яд") || clean.contains("yandex") -> Platform.YANDEX_DIRECT
            clean.contains("вк") || clean.contains("vk") || clean.contains("вконтакте") -> Platform.VK_ADS
            else -> null
        }
    }
}
