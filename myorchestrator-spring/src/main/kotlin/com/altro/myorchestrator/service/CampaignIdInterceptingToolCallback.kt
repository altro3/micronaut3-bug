package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.model.ToolContext
import org.springframework.ai.tool.ToolCallback
import org.springframework.ai.tool.definition.ToolDefinition
import org.springframework.ai.tool.metadata.ToolMetadata
import org.springframework.data.repository.findByIdOrNull
import tools.jackson.databind.json.JsonMapper

class CampaignIdInterceptingToolCallback(
    private val delegate: ToolCallback,
    private val sessionRepository: CampaignSessionRepository,
    private val jsonMapper: JsonMapper
) : ToolCallback {

    private val log = KotlinLogging.logger {}

    override fun getToolDefinition(): ToolDefinition = delegate.toolDefinition
    override fun getToolMetadata(): ToolMetadata = delegate.toolMetadata
    override fun call(toolInput: String): String = delegate.call(toolInput)

    override fun call(toolInput: String, toolContext: ToolContext?): String {
        val result = delegate.call(toolInput, toolContext)

        if (result.isBlank() || result.contains("❌")) return result

        if (delegate.toolDefinition.name() == "initYandexDraft") {
            try {
                val rootNode = jsonMapper.readTree(result)
                val targetNode = if (rootNode.isArray && rootNode.size() > 0) {
                    val firstElement = rootNode.get(0)
                    val escapedJsonText = firstElement.get("text")?.asString()
                    if (!escapedJsonText.isNullOrBlank()) jsonMapper.readTree(escapedJsonText) else firstElement
                } else {
                    rootNode
                }

                if (toolContext != null) {
                    val session = toolContext.context["currentSession"] as? CampaignSession

                    if (session != null) {
                        val idNode = targetNode.get("campaignId")
                        if (idNode != null && idNode.isNumber) {
                            val foundId = idNode.asLong()
                            if (foundId != 0L) {
                                log.info { "🎯 [MCP-Interceptor] Перехвачен первый ответ initYandexDraft. Найдено campaignId=$foundId для сессии ${session.id}" }

                                val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session
                                currentSession.campaignId = foundId
                                sessionRepository.save(currentSession)

                                session.campaignId = foundId
                                log.info { "固 [MCP-Interceptor УСПЕХ] campaignId=$foundId жестко сохранен в Postgres!" }
                            }
                        }

                        val statusNode = targetNode.get("status")
                        if (statusNode?.asString() == "SYNC_ERROR" && targetNode.get("errorAction") != null && !targetNode.get("errorAction").isNull) {
                            log.warn { "⚠️ [MCP-Interceptor] Сбой сессии. Сбрасываю контекст в СУБД..." }
                            val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session
                            currentSession.campaignId = null
                            sessionRepository.save(currentSession)
                            session.campaignId = null
                        }
                    }
                }
            } catch (e: Exception) {
                log.error(e) { "Ошибка асинхронного парсинга ответа от initYandexDraft внутри ToolCallback" }
            }
        }
        return result
    }
}
