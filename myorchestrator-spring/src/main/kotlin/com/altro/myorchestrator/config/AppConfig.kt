package com.altro.myorchestrator.config

import com.altro.common.util.api.config.DefWebConfig
import com.altro.common.util.api.json.JsonUtil
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Primary
import tools.jackson.core.JsonParser
import tools.jackson.core.json.JsonReadFeature.ALLOW_UNQUOTED_PROPERTY_NAMES
import tools.jackson.databind.DeserializationContext
import tools.jackson.databind.JsonNode
import tools.jackson.databind.deser.std.StdDeserializer
import tools.jackson.databind.json.JsonMapper
import tools.jackson.databind.module.SimpleModule

@Configuration
class AppConfig {

    @Primary
    @Bean
    fun jsonMapper(): JsonMapper {
        val module = SimpleModule().apply {
            addDeserializer(Message::class.java, SpringAiMessageDeserializer())
        }
        return JsonUtil.jsonBuilder()
            .enable(ALLOW_UNQUOTED_PROPERTY_NAMES)
            .addModule(module)
            .build()
    }

    class SpringAiMessageDeserializer : StdDeserializer<Message>(Message::class.java) {

        override fun deserialize(p: JsonParser, ctxt: DeserializationContext): Message {
            val node: JsonNode = p.readValueAsTree()

            val typeStr = node.get("messageType")?.asString() ?: "USER"

            return when (typeStr) {
                "USER" -> UserMessage(node.get("content")?.asString() ?: "")
                "ASSISTANT" -> AssistantMessage(node.get("content")?.asString() ?: "")
                "SYSTEM" -> SystemMessage(node.get("content")?.asString() ?: "")
                "TOOL" -> {
                    val toolCallId = node.get("id")?.asString() ?: "default-id"
                    val toolName = node.get("name")?.asString() ?: "unknown-tool"

                    val contentNode = node.get("content")
                    val rawContent = if (contentNode != null && (contentNode.isContainer)) {
                        contentNode.toString()
                    } else {
                        contentNode?.asString() ?: ""
                    }

                    val toolResponse = ToolResponseMessage.ToolResponse(toolCallId, toolName, rawContent)
                    ToolResponseMessage.builder()
                        .responses(listOf(toolResponse))
                        .build()
                }
                else -> UserMessage(node.get("content")?.asString() ?: "")
            }
        }
    }

    @Configuration
    class WebConfig(jsonMapper: JsonMapper) : DefWebConfig(jsonMapper)
}
