package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import java.time.Instant
import java.util.UUID

@Service
class ChatOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val sessionTransactionService: SessionTransactionService,
    private val dynamicAiOrchestrator: DynamicAiOrchestrator
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateChatStream(rq: ChatRq): Flux<String> {
        if (rq.messages.isEmpty()) {
            return Flux.just("👋 Привет! Я твой автономный ИИ-маркетолог AdBroker. Опиши свой бизнес и какую рекламу мы запускаем?")
        }

        val userInput = rq.messages.last().content.trim()

        val sessionIdStr = rq.sessionId
        val parsedSessionId = if (!sessionIdStr.isNullOrBlank()) UUID.fromString(sessionIdStr) else null

        var session: CampaignSession? = null
        if (parsedSessionId != null) {
            session = sessionRepository.findByIdOrNull(parsedSessionId)
        }

        val sessionInitFlux = if (session == null) {
            log.info { "🎯 [ChatOrchestrator] Сессия не найдена или это первый запуск. Инициирую новую долгоживущую ИИ-сессию..." }

            val newSession = CampaignSession(
                id = UUID.randomUUID(),
                updatedAt = Instant.now()
            ).apply { isNewEntity = true }

            val savedSession = sessionTransactionService.saveSessionForce(newSession)
            session = savedSession

            Flux.just("[SESSION_ID:${savedSession.id}]")
        } else {
            log.info { "🎯 [ChatOrchestrator] Сессия успешно найдена по UUID: ${session.id}. Текущая платформа: ${session.platform}. Продолжаю стрим..." }
            Flux.empty()
        }

        val agentResponseFlux = dynamicAiOrchestrator.orchestrateDynamicStream(session!!, userInput)

        return Flux.concat(sessionInitFlux, agentResponseFlux)
            .onErrorResume { error ->
                log.error(error) { "Критический сбой в основном распределителе чата" }
                Flux.just("❌ Критический сбой пайплайна чата: ${error.message}")
            }
            .cache()
    }
}
