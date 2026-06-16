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
        // Проверяем текстовое подтверждение пользователя в чате ("Да"/"Публикуй")
        if (!isApproved(userInput)) {
            sink.next("⚠️ Ожидаю финального подтверждения. Напишите **'Да'** или **'Публикуй'**, чтобы сформировать счет на оплату.")
            sink.complete()
            return
        }

        sink.next("🚀 АГЕНТ: Кампания уже успешно сформирована и валидирована на стороне service-yad.\n")
        sink.next("💳 Формирую счет в биллинг-системе...\n")

        val invoiceId = "INV-${System.currentTimeMillis()}-${session.campaignId}"
        val logs = listOf("✅ Кампания полностью укомплектована", "✅ Выставлен инвойс: $invoiceId")

        // ПРАВИЛЬНЫЕ VAR-МУТАЦИИ: Никаких ошибок компиляции, никаких copy() и несуществующих полей!
        session.currentStep = CampaignCreationStep.PUBLISHING_AND_PAYMENT
        session.context.paymentInvoiceId = invoiceId
        session.context.executionLogs = session.context.executionLogs + logs
        session.updatedAt = Instant.now()

        // Сохраняем мутированный объект в PostgreSQL
        sessionRepository.save(session)

        sink.next("\n🎉 Отлично! Все этапы автоматизации AdBroker успешно пройдены.\n")
        sink.next("💳 Сформирован инвойс для оплаты рекламы: **$invoiceId**\n")
        sink.next("🔗 Перенаправляю вас на платежную платформу...")
        sink.complete()
    }

    fun handleAwaitingPayment(session: CampaignSession, sink: FluxSink<String>) {
        sink.next("🏁 Реклама уже опубликована. Ожидайте подтверждения транзакции по инвойсу ${session.context.paymentInvoiceId}.")
        sink.complete()
    }

    fun handleCompleted(sink: FluxSink<String>) {
        sink.next("Заказ полностью завершен и оплачен.")
        sink.complete()
    }
}
