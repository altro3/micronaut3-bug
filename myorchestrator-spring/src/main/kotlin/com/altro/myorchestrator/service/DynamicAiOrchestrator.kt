package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import tools.jackson.databind.json.JsonMapper
import java.time.Instant

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val sessionTransactionService: SessionTransactionService,
    private val vectorStoreService: VectorStoreService,
    private val platformRouter: PlatformRouter,
    private val mcpTraceParser: McpTraceParser,
    private val agentPromptProvider: AgentPromptProvider,

    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        val currentPlatform = platformRouter.determineTargetPlatform(session, userInput)

        val filterPlatforms = if (currentPlatform == Platform.CHAT) emptyList() else listOf(currentPlatform)
        val moderationRules = vectorStoreService.searchAdvertisingRules(filterPlatforms, userInput)

        val agentSystemPrompt = agentPromptProvider.getSystemPromptForPlatform(currentPlatform, session.campaignId, moderationRules)
        val selectedAgent = when (currentPlatform) {
            Platform.YANDEX_DIRECT -> yandexAgentClient
            Platform.VK_ADS -> vkAgentClient
            Platform.CHAT -> consultantAgentClient
        }

        val historyMessages = mutableListOf<Message>().apply {
            add(SystemMessage(agentSystemPrompt))
            session.context.executionLogs.forEach { jsonStr ->
                try {
                    val msg = mcpTraceParser.deserializeLogToMessage(jsonStr)
                    if (msg.messageType.name != "TOOL") {
                        add(msg)
                    }
                } catch (e: Exception) {
                    log.warn { "Ошибка десериализации истории: ${e.message}" }
                }
            }
        }

        val tempMemory = MessageWindowChatMemory.builder()
            .chatMemoryRepository(InMemoryChatMemoryRepository())
            .build()
        val conversationId = session.id.toString()
        tempMemory.add(conversationId, historyMessages)

        val startCount = historyMessages.size

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента [$currentPlatform]..." }

        var clientPrompt = selectedAgent.prompt()
            .user(userInput)
            .advisors(MessageChatMemoryAdvisor.builder(tempMemory).build())
            .advisors { it.param("chat_memory_conversation_id", conversationId) }

        if (currentPlatform == Platform.YANDEX_DIRECT && session.campaignId != null) {
            log.info { "🔒 [Orchestrator] Инжектирую запрет дубликатов черновиков для ID: ${session.campaignId}" }
            clientPrompt = clientPrompt.system(
                """
                ВНИМАНИЕ! Рекламная кампания УЖЕ успешно создана. 
                Текущий числовой ID кампании в системе равен строго: ${session.campaignId}.
                Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО вызывать инструмент 'initYandexDraft' заново!
                Используй исключительно готовый ID ${session.campaignId} во всех остальных инструментах.
            """.trimIndent()
            )
        }

        val chatResponse = clientPrompt.call().chatResponse()
        val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сформировать ответ."

        val currentSession = sessionRepository.findByIdOrNull(session.id ?: error("ID сессии равен null")) ?: session

        val extractedId = mcpTraceParser.extractCampaignIdFromChatResponse(chatResponse)
        val finalCampaignId = extractedId ?: currentSession.campaignId

        log.info { "🔍 [Orchestrator] Сверка ID кампании: извлечено=$extractedId, итоговый=$finalCampaignId" }

        val nextPlatform = if (currentPlatform == Platform.CHAT && agentPromptProvider.isTriggerTextToStartAutomation(finalResponseText)) {
            if (finalResponseText.contains("Яндекс", ignoreCase = true)) Platform.YANDEX_DIRECT else Platform.VK_ADS
        } else {
            currentPlatform
        }

        val pagedMessages = tempMemory.get(conversationId)
        val newTechnicalMessages = if (pagedMessages.size > startCount) pagedMessages.subList(startCount, pagedMessages.size) else emptyList()
        val serializedNewLogs = mcpTraceParser.serializeMessagesToLogs(newTechnicalMessages)
        currentSession.context.executionLogs += serializedNewLogs

        currentSession.platform = nextPlatform
        currentSession.campaignId = finalCampaignId
        currentSession.updatedAt = Instant.now()

        log.info { "⏳ [Orchestrator] Отправляю плоские поля на физический SQL UPDATE в Postgres..." }
        val saved = sessionTransactionService.saveSessionForce(currentSession)

        session.campaignId = saved.campaignId
        session.platform = saved.platform
        session.context.executionLogs = saved.context.executionLogs

        log.info { "💾 [Orchestrator УСПЕХ] Жесткий UPDATE закоммичен! Платформа в БД: ${saved.platform}, campaignId в БД: ${saved.campaignId}" }

        return Flux.just(finalResponseText)
    }

}
