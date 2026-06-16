package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CreativeGeneratorProcessor
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import tools.jackson.databind.json.JsonMapper
import java.time.Instant

@Component
class CreativeGenerationHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val creativeGeneratorProcessor: CreativeGeneratorProcessor,
    private val mcpClient: McpSyncClient,
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

        var moderationRules = "Соблюдать стандартные лимиты символов."

        // 🌟 ИЗВЛЕКАЕМ РЕАЛЬНЫЕ ПРАВИЛА ИЗ АДАПТЕРА ЧЕРЕЗ MCP ПЕРЕД ГЕНЕРАЦИЕЙ
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                sink.next("📡 Подтягиваю актуальные правила модерации из базы данных адаптера...\n")
                val statusRequest = CallToolRequest("getYandexCampaignStatus", mapOf("campaignId" to yadId), mapOf())
                val statusResponse = mcpClient.callTool(statusRequest)

                val rootNode = jsonMapper.readTree(statusResponse.content.toString())
                // Вытаскиваем правила, которые DictService адаптера бережно сохранил в поле errorMessage или колонку таблицы
                moderationRules = rootNode.path("errorMessage").asString().takeIf { it.isNotBlank() }
                    ?: "Лимит заголовка — 35 символов, лимит текста объявления — 81 символ."
            } catch (e: Exception) {
                log.warn { "Не удалось получить правила из адаптера, используем дефолтные лимиты: ${e.message}" }
            }
        }

        sink.next("⚡ Запускаю Qwen-35B (n_predict=400). Формирую продающие тексты...\n")

        // 1. Генерируем чистый текст через локальный инференс, передавая реальные правила модерации
        val briefContext = session.context.executionLogs.firstOrNull() ?: "Общий бриф"
        val creative = creativeGeneratorProcessor.generate(platform, briefContext, moderationRules)

        // 2. СРАЗУ пушим готовый текст в service-yad по MCP.
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                sink.next("📡 Отправляю сгенерированный текст в service-yad через MCP-инструмент submitYandexCreative...\n")
                // ЖЕЛЕЗОБЕТОННЫЙ ЗАЩИТНЫЙ ЭЛВИС-ОПЕРАТОР ДЛЯ СТРОК С ПЛАТФОРМЫ
                val mcpRequest = CallToolRequest(
                    "submitYandexCreative",
                    mapOf(
                        "campaignId" to yadId,
                        "text" to (creative.bodyText ?: "Специальное предложение от AdBroker")
                    ),
                    mapOf()
                )
                val mcpResponse = mcpClient.callTool(mcpRequest)
                sink.next("✅ Адаптер успешно принял креатив: ${mcpResponse.content}\n\n")
            } catch (e: Exception) {
                sink.next("⚠️ Ошибка синхронизации с адаптером Яндекса: ${e.message}\n")
            }
        }

        // 3. Обновляем var-статус шага в оркестраторе
        session.currentStep = CampaignCreationStep.CREATIVE_GENERATION
        session.updatedAt = Instant.now()
        sessionRepository.save(session)

        // 4. Выводим сгенерированные тексты пользователю в React-чат
        sink.next("✨ **Рекламные тексты успешно сформированы и сохранены:**\n")
        sink.next("\n📢 **ПЛОЩАДКА: [${platform.name}]**\n")
        sink.next("🔹 Заголовок: ${creative.title ?: "Без заголовка"}\n")
        sink.next("🔸 Текст объявления: ${creative.bodyText ?: "Без текста"}\n\n")

        sink.next("🤖 **[Шаг 4/5]**: Проверяем готовность кампании в адаптере и переходим к биллингу? Напишите **'Да'** или **'Публикуй'**.")
        sink.complete()
    }
}
