package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.service.advisor.CampaignSessionIdAdvisor
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.session.Session
import org.springframework.ai.session.SessionEvent
import org.springframework.ai.session.SessionService
import org.springframework.ai.session.advisor.SessionMemoryAdvisor
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import tools.jackson.databind.json.JsonMapper

@Service
class DynamicAiOrchestrator(
    private val sessionService: SessionService,
    private val platformRouter: PlatformRouter,
    private val vectorStoreService: VectorStoreService,
    private val agentPromptProvider: AgentPromptProvider,
    private val jsonMapper: JsonMapper,

    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(aiSession: Session, userInput: String): Flux<String> {
        val conversationId = aiSession.id()

        val savedPlatformStr = aiSession.metadata()["platform"] as? String
        val savedPlatform = savedPlatformStr?.let { Platform.valueOf(it) }

        val currentCampaignId = when (val id = aiSession.metadata()["campaignId"]) {
            is Long -> id
            is Int -> id.toLong()
            is String -> id.toLongOrNull()
            else -> null
        }

        val determinedPlatform = platformRouter.determineTargetPlatform(savedPlatform, userInput)

        val currentRole = when {
            determinedPlatform == Platform.YANDEX_DIRECT || savedPlatform == Platform.YANDEX_DIRECT -> AgentRole.YANDEX_EXPERT
            determinedPlatform == Platform.VK_ADS || savedPlatform == Platform.VK_ADS -> AgentRole.VK_EXPERT
            else -> AgentRole.CONSULTANT
        }

        val selectedAgent = when (currentRole) {
            AgentRole.YANDEX_EXPERT -> yandexAgentClient
            AgentRole.VK_EXPERT -> vkAgentClient
            AgentRole.CONSULTANT -> consultantAgentClient
        }

        val moderationRules = vectorStoreService.searchAdvertisingRules(listOfNotNull(determinedPlatform), userInput)
        val agentSystemPrompt = agentPromptProvider.getSystemPromptForRole(currentRole, moderationRules)

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента в роли [$currentRole] с campaignId=${currentCampaignId ?: 0L}..." }

        val chatResponse = selectedAgent.prompt()
            .system(agentSystemPrompt)
            .user(userInput)
            .advisors { it.param(SessionMemoryAdvisor.SESSION_ID_CONTEXT_KEY, conversationId) }
            .advisors(CampaignSessionIdAdvisor(sessionService, jsonMapper, order = 100))
            .toolContext(
                mapOf(
                    "campaignId" to (currentCampaignId ?: 0L),
                    "sessionId" to conversationId
                )
            )
            .call()
            .chatResponse()

        val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сформировать ответ."

        var finalPlatform = savedPlatform
        var platformChanged = false

        if (finalPlatform == null && determinedPlatform != null) {
            finalPlatform = determinedPlatform
            platformChanged = true
        }

        if (finalPlatform == null && agentPromptProvider.isTriggerTextToStartAutomation(finalResponseText)) {
            finalPlatform = if (finalResponseText.contains("Яндекс", ignoreCase = true)) Platform.YANDEX_DIRECT else Platform.VK_ADS
            platformChanged = true
        }

        if (platformChanged && finalPlatform != null) {
            log.info { "🔄 [Orchestrator] Фиксирую смену платформы на ${finalPlatform.name} через ивент..." }

            sessionService.appendEvent(
                SessionEvent.builder()
                    .sessionId(conversationId)
                    .message(SystemMessage("SYSTEM_EVENT: Platform changed to ${finalPlatform.name}"))
                    .metadata(mapOf("platform" to finalPlatform.name))
                    .build()
            )
        }

        val cleanedResponseText = finalResponseText.replace("\\n", "\n")
            .replace("\r", "")

        return Flux.just(cleanedResponseText)
    }
}
