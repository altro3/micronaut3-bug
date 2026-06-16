package com.altro.myorchestrator.service

import org.springframework.ai.chat.client.ChatClient
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service

@Service
class AiInferenceService(
    chatClientBuilder: ChatClient.Builder,
) {

    private val chatClient by lazy { chatClientBuilder.build() }

    fun analyzeBrief(briefText: String): AnalysisResult {
        return chatClient.prompt()
            .user("Проанализируй следующий бриф, выдели список целевых площадок (YANDEX_DIRECT, VK_ADS) и текстовое описание аудитории. Бриф: $briefText")
            .call()
            .entity(object : ParameterizedTypeReference<AnalysisResult>() {})
            ?: throw IllegalStateException("Ошибка парсинга структурированного ответа ИИ")
    }

    fun generateCreativesJson(systemInstruction: String, userPrompt: String): String {

        return chatClient.prompt()
            .system(systemInstruction)
            .user(userPrompt)
            .call()
            .content()?.trim()
            ?: throw IllegalStateException("Модель вернула пустой ответ при генерации креативов")
    }
}
