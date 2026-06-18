package com.altro.myorchestrator.service

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.stereotype.Component
import tools.jackson.databind.json.JsonMapper

@Component
class McpTraceParser(
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    fun extractCampaignIdFromToolResponses(messages: List<Message>): Long? {
        for (msg in messages) {
            if (msg is ToolResponseMessage) {
                try {
                    val resp = msg.responses.firstOrNull()
                    var rawJson = msg.text ?: ""

                    if (rawJson.isBlank() && resp?.responseData != null) {
                        rawJson = jsonMapper.writeValueAsString(resp.responseData)
                    }

                    if (rawJson.isBlank()) continue

                    if (rawJson.startsWith("\"") && rawJson.endsWith("\"") && rawJson.length > 2) {
                        try {
                            rawJson = jsonMapper.readValue(rawJson, String::class.java)
                        } catch (e: Exception) {
                        }
                    }

                    val rootNode = jsonMapper.readTree(rawJson)
                    val idNode = rootNode.get("id") ?: rootNode.get("campaignId") ?: rootNode.get("campaign_id")

                    if (idNode != null && idNode.isNumber) {
                        return idNode.asLong()
                    }
                } catch (e: Exception) {
                    log.warn { "Не удалось пропарсить JSON-след инструмента для сбора ID: ${e.message}" }
                }
            }
        }
        return null
    }

    fun extractCampaignIdFromTextFallback(text: String): Long? {
        if (text.isBlank()) return null
        val regexes = listOf(
            Regex("""(?:ID|id|идентификатор|кампании|черновика|номером)\s*(?:равен|ориентир|:\s*|=)?\s*(\d+)"""),
            Regex("""\b(\d{2,18})\b""")
        )
        for (regex in regexes) {
            val match = regex.find(text)
            if (match != null) {
                try {
                    return match.groupValues[1].toLong()
                } catch (e: Exception) {
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
