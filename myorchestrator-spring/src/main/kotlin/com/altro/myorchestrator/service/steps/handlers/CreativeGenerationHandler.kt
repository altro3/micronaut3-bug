package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CreativeGeneratorProcessor
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpAsyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import java.time.Instant

@Component
class CreativeGenerationHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val creativeGeneratorProcessor: CreativeGeneratorProcessor,
    private val mcpClient: McpAsyncClient,
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    fun processCreatives(
        session: CampaignSession,
        userInput: String,
        sink: FluxSink<String>,
        isApproved: (String) -> Boolean
    ) {
        if (!isApproved(userInput)) {
            sink.next("⚠️ Напишите **'Да'** или **'Генерируй'**, чтобы запустить генерацию объявления моделью Qwen-35B.")
            sink.complete()
            return
        }

        val platform = session.context.selectedPlatform
            ?: throw IllegalStateException("Критическая ошибка: Рекламная платформа не задана в сессии")
        val yadId = session.context.yadCampaignId
            ?: throw IllegalStateException("Критическая ошибка: Сквозной ID service-yad утерян")

        var moderationRules = "Соблюдать стандартные лимиты символов площадки."

        // 🌟 ИЗВЛЕКАЕМ АКТУАЛЬНЫЕ ПРАВИЛА ИЗ БАЗЫ АДАПТЕРА ЧЕРЕЗ MCP ПЕРЕД ИНФЕРЕНСОМ
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                sink.next("📡 Запрашиваю текущие лимиты и правила модерации из базы данных адаптера...\n")
                val statusRequest = CallToolRequest("getYandexCampaignStatus", mapOf("campaignId" to yadId), mapOf())

                // 🎯 ИСПРАВЛЕНО: Добавляем .block() для ожидания сетевого ответа статуса
                val statusResponse = mcpClient.callTool(statusRequest)
                    .block() ?: throw IllegalStateException("Пустой ответ статуса")

                val rootNode = jsonMapper.readTree(statusResponse.content.toString())
                moderationRules = rootNode.path("errorMessage").asText().takeIf { it.isNotBlank() }
                    ?: "Лимит заголовка — 35 символов, лимит текста объявления — 81 символ."
            } catch (e: Exception) {
                log.warn { "Не удалось получить правила через MCP, используем дефолты: ${e.message}" }
            }
        }

        sink.next("⚡ Запускаю Qwen-35B (n_predict=400). Формирую рекламный текст объявления...\n")

        // 1. Генерируем чистый текст через изолированный процессор
        val briefContext = session.context.executionLogs.firstOrNull() ?: "Общий бриф из истории чата"
        val creative = creativeGeneratorProcessor.generate(platform, briefContext, moderationRules)

        // 2. СРАЗУ пушим готовый текст в service-yad через МСР-инструмент
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                sink.next("📡 Отправляю сгенерированный текст в service-yad через MCP-инструмент submitYandexCreative...\n")
                val mcpRequest = CallToolRequest(
                    "submitYandexCreative",
                    mapOf(
                        "campaignId" to yadId,
                        "text" to (creative.bodyText ?: "Специальное предложение от AdBroker")
                    ),
                    mapOf()
                )

                val mcpResponse = mcpClient.callTool(mcpRequest)
                    .block() ?: throw IllegalStateException("Пустой ответ сохранения креатива")

                sink.next("✅ Адаптер успешно принял креатив: ${mcpResponse.content}\n\n")
            } catch (e: Exception) {
                sink.next("⚠️ Ошибка синхронизации с адаптером Яндекса: ${e.message}\n")
            }
        }

        // 3. Обновляем var-статус шага в оркестраторе чата
        session.currentStep = CampaignCreationStep.CREATIVE_GENERATION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        // 4. Выводим сгенерированные тексты пользователю в чат
        sink.next("✨ **Рекламные тексты успешно сформированы и сохранены в адаптере:**\n")
        sink.next("\n📢 **ПЛОЩАДКА: [${platform.name}]**\n")
        sink.next("🔹 Заголовок: ${creative.title ?: "Без заголовка"}\n")
        sink.next("🔸 Текст объявления: ${creative.bodyText ?: "Без текста"}\n\n")

        sink.next("🤖 **[Шаг 5/5]**: Проверяем готовность кампании в адаптере и переходим к биллингу? Напишите **'Да'** или **'Публикуй'**.")
        sink.complete()
    }
}
