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
import java.time.Instant

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
        val result = delegate.call(toolInput, toolContext) // Сырой JSON от Адаптера

        if (result.isBlank() || result.contains("❌")) return result

        val session = toolContext?.context["currentSession"] as? CampaignSession
        if (session != null) {
            try {
                val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session
                val rootNode = jsonMapper.readTree(result)

                // Вытаскиваем сгенерированный campaignId для плоской связи
                rootNode.get("campaignId")?.let {
                    if (it.isNumber && it.asLong() != 0L) currentSession.campaignId = it.asLong()
                }

                currentSession.updatedAt = Instant.now()

                val saved = sessionRepository.save(currentSession)
                session.campaignId = saved.campaignId

                log.info { "🎯 [Interceptor УСПЕХ] Вся кампания атомарно сохранена в сессию под ID: ${saved.campaignId}" }
            } catch (e: Exception) {
                log.error(e) { "Ошибка авто-сохранения contextData" }
            }
        }
        return result
    }
}
