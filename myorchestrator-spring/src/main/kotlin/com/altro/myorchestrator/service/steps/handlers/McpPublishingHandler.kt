package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.McpPublishingService
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class McpPublishingHandler(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpPublishingService: McpPublishingService
) {

    fun processPublish(session: CampaignSession, userInput: String, sink: FluxSink<String>, isApproved: (String) -> Boolean) {
        if (!isApproved(userInput)) {
            sink.next("⚠️ Ожидаю апрува. Напишите **'Да'** или **'Публикуй'** для отправки данных в сеть.")
            sink.complete()
            return
        }

        val platform = session.context.selectedPlatform ?: throw IllegalStateException("Платформа не найдена")
        val creatives = session.context.generatedCreatives ?: throw IllegalStateException("Креативы не найдены")

        sink.next("🚀 АГЕНТ: Отправляю прямой RPC-запрос в инструмент автоматизации ${platform.name}...\n")

        val publishResult = mcpPublishingService.publish(session.campaignId, listOf(platform), creatives)
        publishResult.logs.forEach { mcpLog -> sink.next("$mcpLog\n") }

        val invoiceId = "INV-${System.currentTimeMillis()}-${session.campaignId}"

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.PUBLISHING_AND_PAYMENT,
                context = session.context.copy(
                    executionLogs = session.context.executionLogs + publishResult.logs,
                    platformCampaignIds = publishResult.campaignIds,
                    paymentInvoiceId = invoiceId
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("\n✅ Объявление успешно выгружено в кабинет!\n")
        sink.next("💳 Сформирован инвойс для перехода к оплате: **$invoiceId**\n")
        sink.next("🔗 Перенаправляю вас на биллинг-платформу...")
        sink.complete()
    }

    fun handleAwaitingPayment(session: CampaignSession, sink: FluxSink<String>) {
        sink.next("🏁 Реклама уже опубликована. Ожидайте подтверждения транзакции по инвойсу ${session.context.paymentInvoiceId}.")
        sink.complete()
    }

    fun handleCompleted(sink: FluxSink<String>) {
        sink.next("Заказ полностью завершен.")
        sink.complete()
    }
}
