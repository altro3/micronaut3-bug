package com.altro.myorchestrator.service.advisor

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClientRequest
import org.springframework.ai.chat.client.ChatClientResponse
import org.springframework.ai.chat.client.advisor.api.CallAdvisor
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.ai.session.SessionEvent
import org.springframework.ai.session.SessionService
import org.springframework.ai.session.advisor.SessionMemoryAdvisor
import tools.jackson.databind.json.JsonMapper

class CampaignSessionIdAdvisor(
    private val sessionService: SessionService,
    private val jsonMapper: JsonMapper,
    private val order: Int = 100
) : CallAdvisor {

    private val log = KotlinLogging.logger {}

    override fun getName(): String = "CampaignSessionIdAdvisor"
    override fun getOrder(): Int = order

    // Возвращаем твой оригинальный, рабочий вариант метода
    override fun adviseCall(rq: ChatClientRequest, chain: CallAdvisorChain): ChatClientResponse {
        val rs = chain.nextCall(rq)
        val conversationId = rq.context[SessionMemoryAdvisor.SESSION_ID_CONTEXT_KEY] as? String ?: return rs
        processMessagesForCampaignId(rq.prompt.instructions, conversationId)
        return rs
    }

    private fun processMessagesForCampaignId(messages: List<org.springframework.ai.chat.messages.Message>, conversationId: String) {
        for (msg in messages) {
            if (msg !is ToolResponseMessage) continue

            for (resp in msg.responses) {
                val rawContent = resp.responseData
                if (rawContent.isBlank() || rawContent.contains("❌")) continue

                try {
                    val rootNode = jsonMapper.readTree(rawContent)
                    val idNode = rootNode.get("campaignId")

                    if (idNode != null && idNode.isNumber) {
                        val foundId = idNode.asLong()

                        if (foundId != 0L) {
                            log.info { "🎯 [CampaignSessionIdAdvisor] Из ответа '${resp.name}' успешно извлечен campaignId: $foundId" }
                            val campaignEvent = SessionEvent.builder()
                                .sessionId(conversationId)
                                .message(SystemMessage("SYSTEM_EVENT: Campaign ID captured: $foundId"))
                                .metadata(mapOf("campaignId" to foundId))
                                .build()


                            sessionService.appendEvent(campaignEvent)
                        }
                    }

                    val statusNode = rootNode.get("status")
                    if (statusNode?.asString() == "SYNC_ERROR" && rootNode.get("errorAction") != null) {
                        log.warn { "⚠️ [CampaignSessionIdAdvisor] Сбой сессии. Генерирую событие сброса..." }

                        val resetEvent = SessionEvent.builder()
                            .sessionId(conversationId)
                            .message(SystemMessage("SYSTEM_EVENT: Campaign context reset due to SYNC_ERROR"))
                            .metadata(mapOf("campaignId" to 0L))
                            .build()

                        sessionService.appendEvent(resetEvent)
                    }

                } catch (_: Exception) {
                    log.trace { "Ответ инструмента '${resp.name}' не является структурой контракта, пропускаем." }
                }
            }
        }
    }
}
