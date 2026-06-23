package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.AgentPrompt.YAD_MESSAGE
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.memory.ChatMemory.CONVERSATION_ID
import org.springframework.ai.chat.memory.ChatMemoryRepository
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers
import java.lang.StringBuilder
import java.time.Instant

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val sessionTransactionService: SessionTransactionService,
    private val platformRouter: PlatformRouter,
    private val chatMemoryRepository: ChatMemoryRepository,

    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        val conversationId = session.id.toString()
        val currentPlatform = platformRouter.determineTargetPlatform(session.platform, userInput)

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

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента: роль [$currentRole] | платформа [$currentPlatform] | сессия $conversationId" }

        val sessionCampaignId = session.campaignId ?: 0L
        val anchoredUserInput = """
        [SYSTEM MEMORY: active_campaign_id=$sessionCampaignId]
        $userInput
    """.trimIndent()

        log.info { "🚀 [Orchestrator] Вызов ИИ-Агента. Фиксированный ID в якоре: $sessionCampaignId" }

        val fullSystemPrompt = """
$YAD_MESSAGE
🎯 ТЕХНИЧЕСКИЙ КОНТЕКСТ СЕССИИ (ID КАМПАНИИ И СЛОТЫ):
ID Кампании в системе: $sessionCampaignId
""".trimIndent()

        return selectedAgent.prompt()
            .system(fullSystemPrompt)
            .user(anchoredUserInput)
            .advisors { it.param(CONVERSATION_ID, conversationId) }
            .toolContext(
                mapOf(
                    "sessionId" to conversationId,
                    "campaignId" to sessionCampaignId,
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
