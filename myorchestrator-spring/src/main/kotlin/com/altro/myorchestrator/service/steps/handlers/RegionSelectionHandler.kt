package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AiInferenceService
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef
import java.time.Instant

@Component
class RegionSelectionHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpSyncClient,
    private val aiInferenceService: AiInferenceService,
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    fun processRegions(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val yadId = session.context.yadCampaignId
        var finalRegionIds = emptyList<String>()

        if (session.context.selectedPlatform == Platform.YANDEX_DIRECT && yadId != null) {
            try {
                sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Сопоставляю города со справочником Яндекса...\n")

                // 🌟 ШАГ 1: ЗАПРАШИВАЕМ СПРАВОЧНИК СТРОГО ЧЕРЕЗ МСР-ИНСТРУМЕНТ
                val regionsRequest = CallToolRequest("getRegionsList", emptyMap(), emptyMap())
                val regionsResponse = mcpClient.callTool(regionsRequest)
                val dictionaryJson = regionsResponse.content.toString()

                // ШАГ 2: Просим локальный ИИ сделать нечеткое сопоставление (Fuzzy Matching)
                val systemInstruction = """
                    Вы — изолированный API-компонент нечеткого поиска. Вам ЗАПРЕЩЕНО общаться с пользователем.
                    Вам на вход дан справочник регионов в формате JSON и текстовый ввод пользователя.
                    Найдите в справочнике регионы, которые упомянул пользователь (учитывайте сокращения, сленг, опечатки, например 'питер' -> 'Санкт-Петербург', 'нск' -> 'Новосибирск').
                    Вы должны вернуть ответ СТРОГО в формате валидного JSON-массива строк, содержащего только ID найденных регионов.
                    НЕ используйте markdown-разметку и кавычки ```json. Начните ответ сразу со знака [.
                    Пример ответа: ["id_1", "id_2"]
                """.trimIndent()

                val userPrompt = """
                    Справочник регионов:
                    $dictionaryJson
                    
                    Ввод пользователя: '$userInput'
                """.trimIndent()

                // Вызов Qwen-35B с жестким n_predict лимитом токенов
                val rawJsonResult = aiInferenceService.generateCreativesJson(systemInstruction, userPrompt)
                finalRegionIds = jsonMapper.readValue(rawJsonResult, jacksonTypeRef<List<String>>())

                if (finalRegionIds.isEmpty()) {
                    sink.next("⚠️ Не удалось распознать указанные города. Пожалуйста, напишите регионы понятнее (например: *Москва, Самара*):")
                    sink.complete()
                    return
                }

                // 🌟 ШАГ 3: ДЕТЕРМИНИРОВАННО ПУШИМ НАЙДЕННЫЕ ID В МСР-ИНСТРУМЕНТ
                sink.next("📡 Отправляю распознанные ID регионов $finalRegionIds в service-yad...\n")
                val mcpRequest = CallToolRequest("bindYandexRegions", mapOf("campaignId" to yadId, "regionIds" to finalRegionIds), mapOf())
                mcpClient.callTool(mcpRequest)

            } catch (e: Exception) {
                log.error(e) { "Ошибка ИИ-сопоставления регионов через MCP" }
                sink.next("⚠️ Ошибка: Не удалось автоматически распознать регионы: ${e.message}\n")
                sink.complete()
                return
            }
        }

        // Чистая мутация var-параметров стейта в PostgreSQL оркестратора
        session.currentStep = CampaignCreationStep.REGION_SELECTION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        sink.next("✅ Регионы таргетинга успешно привязаны по ID.\n\n")
        sink.next("🤖 **[Шаг 2.5/5]**: Теперь укажите возрастные ограничения для вашей аудитории (например: *18+, 0+*):")
        sink.complete()
    }
}
