package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.ChatRq
import com.altro.myorchestrator.api.dto.InitSessionRs
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono
import reactor.core.scheduler.Schedulers
import java.time.Instant
import java.util.UUID

@Service
class ChatOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val dynamicAiOrchestrator: DynamicAiOrchestrator
) {

    private val log = KotlinLogging.logger {}

    fun createNewSession(): Mono<InitSessionRs> {
        return Mono.fromCallable {
            val newSession = CampaignSession(
                id = UUID.randomUUID(),
                updatedAt = Instant.now()
            )
            val saved = sessionRepository.save(newSession)
            log.info { "🎯 [Оркестратор] Сгенерирована и сохранена новая сессия в БД. ID: ${saved.id}" }
            InitSessionRs(sessionId = saved.id)
        }.subscribeOn(Schedulers.boundedElastic())
    }

    fun orchestrateChatStream(rq: ChatRq): Flux<String> =
        Flux.create { sink ->
            try {
                if (rq.messages.isEmpty()) {
                    sink.next("👋 Привет! Я твой автономный ИИ-маркетолог AdBroker. Опишите ваш бизнес и какую рекламу мы запускаем?")
                    sink.complete()
                    return@create
                }

                val userInput = rq.messages.last().content.trim()
                val session = sessionRepository.findByIdOrNull(rq.sessionId)

                if (session == null) {
                    log.error { "❌ [Оркестратор] Фронтенд прислал несуществующий sessionId: ${rq.sessionId}" }
                    sink.next("❌ Ошибка: Сессия диалога не найдена на сервере. Пожалуйста, обновите чат.")
                    sink.complete()
                    return@create
                }

                if (session.context.rawBriefText == null) {
                    log.info { "🎯 [Оркестратор] Первая реплика для сессии ${session.id}. Фиксирую бриф..." }
                    session.context.rawBriefText = userInput
                }

                log.info { "🎯 [Оркестратор] Сессия успешно возобновлена (ID: ${session.id}, Платформа: ${session.context.platform}). Продолжаю стрим..." }

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
