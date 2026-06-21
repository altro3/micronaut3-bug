package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AgentPromptProvider.getDynamicTechnicalContext
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.memory.ChatMemory.CONVERSATION_ID
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers
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
            else -> consultantAgentClient
        }
        val dynamicQuadrantContext = getDynamicTechnicalContext(
            role = currentRole,
            campaignId = session.campaignId,
            moderationRules = moderationRules
        )

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента: роль [$currentRole] | платформа [$currentPlatform] | сессия $conversationId" }

        return selectedAgent.prompt()
            .user(userInput)
            .system { systemSpec ->
                systemSpec.param("moderationRules", dynamicQuadrantContext)
            }
            .advisors { it.param(CONVERSATION_ID, conversationId) }
            .toolContext(
                mapOf(
                    "sessionId" to conversationId,
                    "campaignId" to (session.campaignId ?: 0L),
                    "currentSession" to session,
                )
            )
            .stream()
            .content()
            .doOnComplete {
                val dbSession = sessionRepository.findByIdOrNull(session.id) ?: session

                dbSession.platform = currentPlatform
                dbSession.updatedAt = Instant.now()

                log.info { "⏳ [Orchestrator] Фиксация сессии. Платформа: ${dbSession.platform}" }
                val saved = sessionTransactionService.saveSessionForce(dbSession)

                session.campaignId = saved.campaignId
                session.platform = saved.platform
            }
            .publishOn(Schedulers.boundedElastic())
            .onErrorResume { error ->
                log.error(error) { "Ошибка стриминга" }
                Flux.just("\n❌ [Ошибка]: ${error.message}")
            }
    }
}
