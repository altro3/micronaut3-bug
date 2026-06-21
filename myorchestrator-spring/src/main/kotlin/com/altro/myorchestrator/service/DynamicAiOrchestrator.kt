package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AgentPromptProvider.getDynamicTechnicalContext
import com.altro.myorchestrator.service.AgentPromptProvider.isTriggerTextToStartAutomation
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.memory.ChatMemory
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import java.time.Instant

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val sessionTransactionService: SessionTransactionService,
    private val vectorStoreService: VectorStoreService,
    private val platformRouter: PlatformRouter,

    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        val conversationId = session.id.toString()
        val currentPlatform = platformRouter.determineTargetPlatform(session.platform, userInput)
        val moderationRules = vectorStoreService.searchAdvertisingRules(listOfNotNull(currentPlatform), userInput)

        val currentRole = when (currentPlatform) {
            Platform.YANDEX_DIRECT -> AgentRole.YANDEX_EXPERT
            Platform.VK_ADS -> AgentRole.VK_EXPERT
            else -> AgentRole.CONSULTANT
        }
        val selectedAgent = when (currentRole) {
            AgentRole.YANDEX_EXPERT -> yandexAgentClient
            AgentRole.VK_EXPERT -> vkAgentClient
            AgentRole.CONSULTANT -> consultantAgentClient
        }
        val dynamicQuadrantContext = getDynamicTechnicalContext(
            role = currentRole,
            campaignId = session.campaignId,
            moderationRules = moderationRules
        )

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента: роль [$currentRole] | платформа [$currentPlatform] | сессия $conversationId" }

        val chatResponse = selectedAgent.prompt()
            .user(userInput)
            .system(dynamicQuadrantContext)
            .toolContext(
                mapOf(
                    ChatMemory.CONVERSATION_ID to conversationId,
                    "sessionId" to conversationId,
                    "campaignId" to (session.campaignId ?: 0L),
                    "currentSession" to session,
                )
            )
            .call()
            .chatResponse()
        val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session
        val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сформировать ответ."

        val nextPlatform = if (currentPlatform == null && isTriggerTextToStartAutomation(finalResponseText)) {
            if (finalResponseText.contains("Яндекс", ignoreCase = true)) Platform.YANDEX_DIRECT else Platform.VK_ADS
        } else {
            currentPlatform
        }

        currentSession.platform = nextPlatform
        currentSession.updatedAt = Instant.now()

        log.info { "⏳ [Orchestrator] Отправляю плоские поля на физический SQL UPDATE в Postgres..." }
        val saved = sessionTransactionService.saveSessionForce(currentSession)

        session.campaignId = saved.campaignId
        session.platform = saved.platform

        log.info { "💾 [Orchestrator УСПЕХ] Жесткий UPDATE закоммичен! Платформа в БД: ${saved.platform}, campaignId в БД: ${saved.campaignId}" }

        return Flux.just(finalResponseText)
    }
}
