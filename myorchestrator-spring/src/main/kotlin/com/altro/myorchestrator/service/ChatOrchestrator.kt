package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import java.time.Instant

@Service
class ChatOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val dynamicAiOrchestrator: DynamicAiOrchestrator
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateChatStream(rq: ChatRq): Flux<String> =
        Flux.create { sink ->
            try {
                if (rq.messages.isEmpty()) {
                    sink.next("👋 Привет! Я твой автономный ИИ-маркетолог AdBroker. Опишите ваш бизнес и какую рекламу мы запускаем?")
                    sink.complete()
                    return@create
                }

                val firstMessage = rq.messages.first().content.trim()
                val userInput = rq.messages.last().content.trim()

                var session = sessionRepository.findByFirstMessage(firstMessage)

                if (session == null) {
                    log.info { "🎯 [Оркестратор] Первая реплика. Инициирую новую долгоживущую ИИ-сессию чата..." }

                    val newSession = CampaignSession(
                        updatedAt = Instant.now()
                    ).apply {
                        context.rawBriefText = firstMessage
                        context.executionLogs = listOf(userInput)
                    }

                    session = sessionRepository.save(newSession)
                } else {
                    log.info { "🎯 [Оркестратор] Сессия найдена (ID сессии: ${session.id}, ID Яндекса: ${session.context.campaignId}). Продолжаю стрим..." }
                }

                dynamicAiOrchestrator.orchestrateDynamicStream(session, userInput)
                    .subscribe(
                        { chunk -> sink.next(chunk) },
                        { error -> sink.error(error) },
                        { sink.complete() }
                    )

            } catch (e: Exception) {
                log.error(e) { "Критический сбой в основном распределителе чата" }
                sink.next("❌ Критический сбой пайплайна чата: ${e.message}")
                sink.error(e)
            }
        }
}
