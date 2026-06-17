package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.modelcontextprotocol.client.McpSyncClient
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider // ОФИЦИАЛЬНЫЙ КЛАСС-МОСТ
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpSyncClient,
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStoreService: VectorStoreService,
) {

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        session.currentStep = CampaignCreationStep.AI_DYNAMIC_COLLECTING

        val moderationRules = vectorStoreService.searchAdvertisingRules(emptyList())

        val systemPrompt = """
            Ты — ИИ-оркестратор. Твоя цель — настроить кампанию в Яндекс.Директ через доступные MCP-инструменты.
            Извлекай ГЕО, возраст, тексты и сразу вызывай тулы. Не мучай юзера пошаговыми вопросами, если он дал всё в одной фразе.
            Если нет ID кампании — начни с 'initYandexDraft'. Если юзер назвал регион словом — сначала найди его ID через 'searchYandexRegions'.
            Правила сетей: $moderationRules
        """.trimIndent()

        val historyMessages = mutableListOf<Message>().apply {
            add(SystemMessage(systemPrompt))
            session.context.executionLogs.forEachIndexed { index, log ->
                if (index % 2 == 0) add(UserMessage(log)) else add(AssistantMessage(log))
            }
            add(UserMessage(userInput))
        }

        val toolProvider = SyncMcpToolCallbackProvider.builder()
            .addMcpClient(mcpClient)
            .build()

        val stringBuffer = StringBuilder()

        return chatClientBuilder.build()
            .prompt()
            .messages(historyMessages)
            // Передаем провайдер. Он сам заберет у service-yad список тулов и свяжет их с LLM!
            .tools(toolProvider)
            .stream()
            .content()
            .doOnNext { stringBuffer.append(it) }
            .doOnComplete {
                val updatedLogs = session.context.executionLogs.toMutableList().apply {
                    add(userInput)
                    add(stringBuffer.toString())
                }
                session.context.executionLogs = updatedLogs
                sessionRepository.save(session)
            }
            .subscribeOn(Schedulers.boundedElastic())
    }
}
