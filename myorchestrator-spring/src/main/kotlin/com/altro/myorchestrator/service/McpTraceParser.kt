package com.altro.myorchestrator.service

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.chat.model.ChatResponse
import org.springframework.stereotype.Component
import tools.jackson.databind.json.JsonMapper

@Component
class McpTraceParser(
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    fun extractCampaignIdFromChatResponse(chatResponse: ChatResponse?): Long? {
        if (chatResponse == null) return null

        val generations = chatResponse.results ?: emptyList()
        for (generation in generations) {
            val assistantMsg = generation.output

            if (assistantMsg != null && assistantMsg.hasToolCalls()) {
                for (toolCall in assistantMsg.toolCalls) {

                    if (toolCall.name == "initYandexDraft") {
                        try {
                            val rawJson = toolCall.arguments ?: ""
                            if (rawJson.isBlank()) continue

                            val rootNode = jsonMapper.readTree(rawJson)

                            val idNode = rootNode.get("id") ?: rootNode.get("campaignId") ?: rootNode.get("campaign_id")
                            if (idNode != null && idNode.isNumber) {
                                val foundId = idNode.asLong()
                                log.info { "🎯 [McpTraceParser] Извлечен валидный ID кампании напрямую из ToolCall: $foundId" }
                                return foundId
                            }
                        } catch (e: Exception) {
                            log.warn { "Ошибка парсинга JSON внутри toolCall: ${e.message}" }
                        }
                    }
                }
            }
        }
        return null
    }

    fun serializeMessagesToLogs(messages: List<Message>): List<String> {
        return messages.map { msg ->
            val dto = when (msg) {
                is ToolResponseMessage -> {
                    val resp = msg.responses.firstOrNull()
                    val cleanContent = if (resp?.responseData != null) jsonMapper.writeValueAsString(resp.responseData) else msg.text ?: ""
                    PersistentMessageDto("TOOL", cleanContent, resp?.id, resp?.name)
                }

                else -> PersistentMessageDto(msg.messageType.name, msg.text ?: "")
            }
            jsonMapper.writeValueAsString(dto)
        }
    }

    fun deserializeLogToMessage(jsonStr: String): Message {
        val dto = jsonMapper.readValue(jsonStr, PersistentMessageDto::class.java)
        return when (dto.messageType) {
            "USER" -> UserMessage(dto.content)
            "ASSISTANT" -> AssistantMessage(dto.content)
            "SYSTEM" -> SystemMessage(dto.content)
            "TOOL" -> {
                val toolResponse = ToolResponseMessage.ToolResponse(
                    dto.id ?: "default-id",
                    dto.name ?: "unknown-tool",
                    dto.content
                )
                ToolResponseMessage.builder().responses(listOf(toolResponse)).build()
            }

            else -> UserMessage(dto.content)
        }
    }

    data class PersistentMessageDto(
        val messageType: String,
        val content: String,
        val id: String? = null,
        val name: String? = null
    )
}
