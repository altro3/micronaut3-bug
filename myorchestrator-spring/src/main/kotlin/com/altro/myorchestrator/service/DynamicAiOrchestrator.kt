package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.mcp.AsyncMcpToolCallbackProvider
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val toolProvider: AsyncMcpToolCallbackProvider, // Твой сетевой клиент
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStoreService: VectorStoreService,
) {

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        session.currentStep = CampaignCreationStep.AI_DYNAMIC_COLLECTING

        val currentPlatforms = listOfNotNull(session.context.platform)
        val moderationRules = vectorStoreService.searchAdvertisingRules(currentPlatforms, userInput)

        val systemPrompt = """
            Ты — ИИ-оркестратор. Твоя цель — настроить кампанию в Яндекс.Директ через доступные MCP-инструменты.
            Извлекай ГЕО, возраст, тексты и сразу вызывай тулы по сети. Если нет ID кампании — начни с 'initYandexDraft'.
            Правила сетей: $moderationRules
        """.trimIndent()

        val historyMessages = mutableListOf<Message>().apply {
            add(SystemMessage(systemPrompt))
            session.context.executionLogs.forEachIndexed { index, log ->
                if (index % 2 == 0) add(UserMessage(log)) else add(AssistantMessage(log))
            }
            add(UserMessage(userInput))
        }

        val stringBuffer = StringBuilder()

        return chatClientBuilder.build()
            .prompt()
            .messages(historyMessages)
            .tools(toolProvider)
            .stream()
            .chatResponse()
            .map { response ->
                val chunk = response.result?.output?.text ?: ""
                stringBuffer.append(chunk)
                chunk
            }
            .doOnComplete {
                val finalResponseText = stringBuffer.toString()
                val updatedLogs = session.context.executionLogs.toMutableList().apply {
                    add(userInput)
                    add(finalResponseText)
                }

                // Ищем ID черновика, который ИИ получил из MCP и залогировал/вывел
                val extractedId = regexFindId(finalResponseText)
                if (extractedId != null) {
                    session.context.campaignId = extractedId
                }

                session.context.executionLogs = updatedLogs
                sessionRepository.save(session)
            }
            .subscribeOn(Schedulers.boundedElastic())
    }

    private fun regexFindId(text: String): Long? {
        val regex = "\"id\"\\s*:\\s*(\\d+)".toRegex()
        val matchResult = regex.find(text)
        return matchResult?.groupValues?.get(1)?.toLongOrNull()
    }
}
