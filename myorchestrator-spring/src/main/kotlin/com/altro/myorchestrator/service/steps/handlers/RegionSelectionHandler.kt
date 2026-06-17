package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AiInferenceService
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpAsyncClient // 🎯 ИСПРАВЛЕНО: Инжектим асинхронный клиент
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef
import java.time.Instant

@Component
class RegionSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpAsyncClient, // 🎯 ИСПРАВЛЕНО: Меняем тип на McpAsyncClient
    private val aiInferenceService: AiInferenceService,
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    fun processRegions(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val campaignId = session.context.campaignId
        var finalRegionIds = emptyList<String>()

        if (session.context.platform == Platform.YANDEX_DIRECT && campaignId != null) {
            try {
                sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Сопоставляю города со справочником Яндекса...\n")

                // Получаем синхронную обертку, чтобы не переписывать весь метод на реактивный стиль
                val regionsRequest = CallToolRequest(
                    "searchYandexRegions",
                    mapOf("query" to userInput.trim()),
                    mapOf()
                )
                val regionsResponse = mcpClient.callTool(regionsRequest)
                    .block() ?: throw IllegalStateException("MCP-сервер вернул пустой ответ при поиске регионов")

                val dictionaryJson = regionsResponse.content.toString()

                // ШАГ 2: Просим локальный ИИ сделать нечеткое сопоставление (Fuzzy Matching) среди отфильтрованных строк
                val systemInstruction = """
                    Вы — изолированный API-компонент нечеткого поиска. Вам ЗАПРЕЩЕНО общаться с пользователем.
                    Вам на вход дан справочник регионов в формате JSON и текстовый ввод пользователя.
                    Найдите в справочнике регионы, которые упомянул пользователь.
                    Вы должны вернуть ответ СТРОГО в формате валидного JSON-массива строк, содержащего только ID найденных регионов.
                    НЕ используйте markdown-разметку. Начните ответ сразу со знака [.
                    Пример ответа: ["id_1", "id_2"]
                """.trimIndent()

                val userPrompt = """
                    Справочник регионов:
                    $dictionaryJson
                    
                    Ввод пользователя: '$userInput'
                """.trimIndent()

                val rawJsonResult = aiInferenceService.generateCreativesJson(systemInstruction, userPrompt)
                finalRegionIds = jsonMapper.readValue(rawJsonResult, jacksonTypeRef<List<String>>())

                if (finalRegionIds.isEmpty()) {
                    sink.next("⚠️ Не удалось распознать указанные города. Пожалуйста, напишите регионы понятнее (например: *Москва, Самара*):")
                    sink.complete()
                    return
                }

                // ШАГ 3: ДЕТЕРМИНИРОВАННО ПУШИМ НАЙДЕННЫЕ ID В МСР-ИНСТРУМЕНТ
                sink.next("📡 Отправляю распознанные ID регионов $finalRegionIds в service-yad...\n")
                val mcpRequest = CallToolRequest("bindYandexRegions", mapOf("campaignId" to campaignId, "regionIds" to finalRegionIds), mapOf())
                mcpClient.callTool(mcpRequest).block()

            } catch (e: Exception) {
                log.error(e) { "Ошибка ИИ-сопоставления регионов через MCP" }
                sink.next("⚠️ Ошибка: Не удалось автоматически распознать регионы: ${e.message}\n")
                sink.complete()
                return
            }
        }

        session.currentStep = CampaignCreationStep.REGION_SELECTION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("✅ Регионы таргетинга успешно привязаны по ID.\n\n")
        sink.next("🤖 **[Шаг 2.5/5]**: Теперь укажите возрастные ограничения для вашей аудитории (например: *18+, 0+*):")
        sink.complete()
    }
}
