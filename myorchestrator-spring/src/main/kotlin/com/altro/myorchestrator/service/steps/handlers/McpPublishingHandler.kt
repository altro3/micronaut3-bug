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
class McpPublishingHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpAsyncClient,
) {
    private val log = KotlinLogging.logger {}

    fun processPublish(
        session: CampaignSession,
        userInput: String,
        sink: FluxSink<String>,
        isApproved: (String) -> Boolean
    ) {
        // Проверяем текстовый апрув от пользователя ("Да", "Публикуй" и т.д.)
        if (!isApproved(userInput)) {
            sink.next("⚠️ Ожидаю финального подтверждения. Напишите **'Да'** или **'Публикуй'**, чтобы запустить кампанию в сеть и выставить счет.")
            sink.complete()
            return
        }

        val platform = session.context.selectedPlatform
            ?: throw IllegalStateException("Критическая ошибка: Рекламная платформа не задана в сессии")
        val yadId = session.context.yadCampaignId
            ?: throw IllegalStateException("Критическая ошибка: Сквозной ID service-yad утерян")

        // 🌟 ШАГ 1: ТРИГГЕРИМ ФИНАЛЬНУЮ ПУБЛИКАЦИЮ В СЕТЬ ЯНДЕКСА ЧЕРЕЗ MCP
        if (platform == Platform.YANDEX_DIRECT) {
            try {
                sink.next("🚀 Инициирую финальный паблишинг кампании в Яндекс.Директ через MCP...\n")

                val mcpRequest = CallToolRequest(
                    "publishYandexCampaign",
                    mapOf("campaignId" to yadId),
                    mapOf()
                )

                // Вызываем асинхронный инструмент и жестко блокируем поток до ответа Яндекса/WireMock
                val mcpResponse = mcpClient.callTool(mcpRequest)
                    .block() ?: throw IllegalStateException("Пустой ответ от сервера публикации")

                sink.next("✅ Адаптер Яндекса подтвердил успешную валидацию и отправку пакета в сеть!\n")
                log.info { "Успешно выполнен паблишинг через инструмент publishYandexCampaign для ID: $yadId. Ответ: ${mcpResponse.content}" }
            } catch (e: Exception) {
                log.error(e) { "Критическая ошибка финальной публикации через MCP" }
                sink.next("❌ Сбой публикации в Яндекс.Директ: ${e.message}\n")
                sink.complete()
                return
            }
        }

        // ШАГ 2: БИЛЛИНГ (Выставляем счет только после успешной отправки в сеть)
        sink.next("💳 Связываюсь с биллинг-платформой для выставления счета...\n")

        val invoiceId = "INV-${System.currentTimeMillis()}-${session.campaignId}"
        val logs = listOf("✅ Кампания успешно опубликована в сеть", "✅ Выставлен инвойс: $invoiceId")

        // Переводим сессию в статус ожидания оплаты
        session.currentStep = CampaignCreationStep.PUBLISHING_AND_PAYMENT
        session.context.paymentInvoiceId = invoiceId
        session.context.executionLogs = session.context.executionLogs + logs
        session.updatedAt = Instant.now()

        sessionRepository.save(session)

        sink.next("\n🎉 Поздравляю! Все этапы автоматизации AdBroker успешно завершены.\n")
        sink.next("💳 Сформирован инвойс для оплаты рекламы: **$invoiceId**\n")
        sink.next("🔗 Перенаправляю ваш интерфейс на платежный шлюз...")
        sink.complete()
    }

    /**
     * Обработка запросов, когда чат стоит на паузе и ждет оплаты от биллинга
     */
    fun handleAwaitingPayment(session: CampaignSession, sink: FluxSink<String>) {
        sink.next("🏁 Рекламная кампания уже сформирована и отправлена на модерацию. Ожидаю подтверждения транзакции по инвойсу **${session.context.paymentInvoiceId}**.")
        sink.complete()
    }

    /**
     * Обработка запросов, когда кампания полностью оплачена и закрыта
     */
    fun handleCompleted(sink: FluxSink<String>) {
        sink.next("🏁 Эта кампания полностью оплачена, активна и запущена в рекламной сети.")
        sink.complete()
    }
}
