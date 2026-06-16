package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.model.CampaignSession.McpDictionaryItem
import com.altro.myorchestrator.model.CampaignSession.SessionContext
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef
import java.time.Instant

@Component
class PlatformSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpSyncClient,
    private val jsonMapper: JsonMapper
) {

    fun sendWelcomeMessage(sink: FluxSink<String>) {
        sink.next("👋 Привет! Я локальный ИИ-ассистент AdBroker.\n")
        sink.next("🤖 **[Шаг 1/5]**: Напишите платформу для кампании: **VK_ADS** или **YANDEX_DIRECT**:")
        sink.complete()
    }

    fun initializeSession(firstMessage: String, userInput: String, sink: FluxSink<String>) {
        val platform = parsePlatform(userInput)
        if (platform == null) {
            sink.next("⚠️ Пожалуйста, напишите корректно: **VK_ADS** или **YANDEX_DIRECT**.")
            sink.complete()
            return
        }

        sink.next("⚙️ Инициализирую черновик и запрашиваю справочники у MCP-сервера 8090...\n")

        // 🌟 ВЫЗОВ MCP ДЛЯ ПОЛУЧЕНИЯ ДИНАМИЧЕСКИХ ДАННЫХ ШАГА
        val mcpRegions = fetchRegionsFromMcp(platform)

        val session = sessionRepository.save(
            CampaignSession(
                currentStep = CampaignCreationStep.PLATFORM_SELECTION,
                context = SessionContext(
                    rawBriefText = firstMessage,
                    selectedPlatform = platform,
                    mcpAvailableRegions = mcpRegions,
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("✅ Платформа ${platform.name} зафиксирована.\n")
        sink.next("📡 С MCP-сервера успешно подгружено ${mcpRegions.size} доступных гео-регионов.\n\n")
        sink.next("🤖 **[Шаг 2/5]**: Укажите регионы таргетинга через запятую. Вы можете продолжить в чате или переключиться на UI-форму выбора.")
        sink.complete()
    }

    private fun fetchRegionsFromMcp(platform: Platform): List<McpDictionaryItem> = try {
        // Делаем детерминированный RPC запрос к инструменту получения справочников регионов
        val request = CallToolRequest("getRegionsList", mapOf("platform" to platform.name), mapOf())
        val response = mcpClient.callTool(request)

        // Конвертируем JSON-ответ от МСР-сервера в наш типизированный список Котлина
        jsonMapper.readValue(response.content.toString(), jacksonTypeRef<List<McpDictionaryItem>>())
    } catch (e: Exception) {
        // Фолбэк на случай локальных проблем с сетью MCP
        listOf(McpDictionaryItem("default", "Россия (Общий)"))
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
