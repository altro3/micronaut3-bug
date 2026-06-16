package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class McpPublishingHandler(
    private val sessionRepository: CampaignSessionRepository
) {

    fun processPublish(
        session: CampaignSession,
        userInput: String,
        sink: FluxSink<String>,
        isApproved: (String) -> Boolean
    ) {
        // Проверяем текстовый апрув от пользователя ("Да", "Публикуй" и т.д.)
        if (!isApproved(userInput)) {
            sink.next("⚠️ Ожидаю финального подтверждения. Напишите **'Да'** или **'Публикуй'**, чтобы выставить счет на оплату кампании.")
            sink.complete()
            return
        }

        sink.next("🚀 АГЕНТ: Рекламная кампания полностью сформирована и валидирована в service-yad.\n")
        sink.next("💳 Связываюсь с биллинг-платформой для выставления счета...\n")

        // Генерируем сквозной инвойс
        val invoiceId = "INV-${System.currentTimeMillis()}-${session.campaignId}"
        val logs = listOf("✅ Кампания укомплектована", "✅ Выставлен инвойс: $invoiceId")

        session.currentStep = CampaignCreationStep.PUBLISHING_AND_PAYMENT
        session.context.paymentInvoiceId = invoiceId
        session.context.executionLogs = session.context.executionLogs + logs
        session.updatedAt = Instant.now()

        // Атомарно сохраняем измененный стейт сессии в PostgreSQL
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
        sink.next("🏁 Рекламная кампания уже сформирована. Ожидаю подтверждения транзакции по инвойсу **${session.context.paymentInvoiceId}**.")
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
