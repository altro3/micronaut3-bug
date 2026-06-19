package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatRq
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.session.CreateSessionRequest
import org.springframework.ai.session.SessionService
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import java.util.UUID

@Service
class ChatOrchestrator(
    private val sessionService: SessionService, // КРИТИЧНО: добавляем сервис сессий либы
    private val dynamicAiOrchestrator: DynamicAiOrchestrator
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateChatStream(rq: ChatRq): Flux<String> =
        Flux.create { sink ->
            try {
                if (rq.messages.isEmpty()) {
                    sink.next("👋 Привет! Я твой автономный ИИ-маркетолог AdBroker. Опиши свой бизнес и какую рекламу мы запускаем?")
                    sink.complete()
                    return@create
                }

                val userInput = rq.messages.last().content.trim()

                val finalSession = rq.sessionId?.let {
                    val existedSession = sessionService.findById(it)
                    log.info { "🎯 [ChatOrchestrator] Продолжаю диалог в рамках существующей сессии UUID: $it" }
                    existedSession
                } ?: run {
                    val newId = UUID.randomUUID()
                    log.info { "🎯 [ChatOrchestrator] Сессия не найдена или это первый запуск. Инициирую новую ИИ-сессию в СУБД: $newId" }

                    val newSession = sessionService.create(
                        CreateSessionRequest.builder()
                            .id(newId.toString())
                            .userId("default_user")
                            .build()
                    )

                    sink.next("[SESSION_ID:$newId]")
                    newSession
                }

                dynamicAiOrchestrator.orchestrateDynamicStream(finalSession, userInput)
                    .subscribe(
                        { chunk -> sink.next(chunk) },
                        { error ->
                            log.error(error) { "Ошибка внутри реактивного стрима агента" }
                            sink.error(error)
                        },
                        { sink.complete() }
                    )

            } catch (e: Exception) {
                log.error(e) { "Критический сбой в основном распределителе чата" }
                sink.next("❌ Критический сбой пайплайна чата: ${e.message}")
                sink.error(e)
            }
        }
}
