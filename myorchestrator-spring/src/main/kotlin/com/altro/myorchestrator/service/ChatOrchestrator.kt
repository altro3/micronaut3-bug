package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatMessageDto
import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.api.dto.chat.ChatRs
import com.altro.myorchestrator.api.dto.chat.ChatRs.ChatChoice
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.stereotype.Service
import java.util.UUID

@Service
class ChatOrchestrator(
    chatClientBuilder: ChatClient.Builder
) {
    private val chatClient = chatClientBuilder.build()

    fun orchestrateChat(request: ChatRq): ChatRs {
        val aiText = processChat(request.messages)

        return ChatRs(
            id = "$ID_PREFIX${UUID.randomUUID()}",
            `object` = OBJECT_TYPE_COMPLETION,
            created = System.currentTimeMillis() / 1000,
            model = request.model,
            choices = listOf(
                ChatChoice(
                    index = 0,
                    message = ChatMessageDto(role = ASSISTANT_ROLE, content = aiText),
                    finish_reason = FINISH_REASON_STOP
                )
            )
        )
    }

    private fun processChat(messages: List<ChatMessageDto>): String {
        var hasSystemPrompt = false
        val springAiMessages = ArrayList<Message>(messages.size + 1)

        for (msg in messages) {
            val role = msg.role.lowercase()
            if (role == SYSTEM_ROLE) {
                hasSystemPrompt = true
            }

            val aiMessage = when (role) {
                SYSTEM_ROLE -> SystemMessage(msg.content)
                ASSISTANT_ROLE -> AssistantMessage(msg.content)
                else -> UserMessage(msg.content)
            }
            springAiMessages.add(aiMessage)
        }

        if (!hasSystemPrompt) {
            springAiMessages.add(0, SystemMessage(SYSTEM_PROMPT))
        }

        val rs = chatClient.prompt()
            .messages(springAiMessages)
            .call()
            .content()

        return rs ?: ERROR_RESPONSE
    }

    companion object {
        private const val SYSTEM_ROLE = "system"
        private const val ASSISTANT_ROLE = "assistant"
        private const val FINISH_REASON_STOP = "stop"
        private const val OBJECT_TYPE_COMPLETION = "chat.completion"
        private const val ID_PREFIX = "chatcmpl-"
        private const val SYSTEM_PROMPT = "Ты — умный ассистент рекламного агрегатора. Твоя цель — помочь пользователю сформулировать бриф для будущей рекламы."
        private const val ERROR_RESPONSE = "Извините, не удалось получить ответ от модели."
    }
}
