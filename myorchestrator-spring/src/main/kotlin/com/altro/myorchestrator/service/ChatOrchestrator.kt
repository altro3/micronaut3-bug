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

    fun orchestrateChatStream(rq: ChatRq): Flux<String> =
        Flux.create { sink ->
            try {
                if (rq.messages.isEmpty()) {
                    sink.next("👋 Привет! Я твой автономный ИИ-маркетолог AdBroker. Опиши свой бизнес и какую рекламу мы запускаем?")
                    sink.complete()
                    return@create
                }

                val userInput = rq.messages.last().content.trim()

                val sessionIdStr = rq.sessionId
                val parsedSessionId = if (!sessionIdStr.isNullOrBlank()) UUID.fromString(sessionIdStr) else null

                var session: CampaignSession? = null
                if (parsedSessionId != null) {
                    session = sessionRepository.findByIdOrNull(parsedSessionId)
                }

                if (session == null) {
                    log.info { "🎯 [Оркестратор] Сессия не найдена или это первый запуск. Инициирую новую долгоживущую ИИ-сессию..." }

                    val newSession = CampaignSession(
                        id = UUID.randomUUID(), // Нативная генерация UUID (или оставь null, если Postgres генерирует сам через Persistable)
                        updatedAt = Instant.now()
                    ).apply {
                        // Фиксируем стартовое описание бизнеса в контекст
                        context.rawBriefText = rq.messages.first().content.trim()
                        // В историю укладываем первое пользовательское сообщение (без системных промптов роутера)
                        context.executionLogs = emptyList()
                        isNewEntity = true
                    }

                    // Сохраняем сессию, чтобы получить валитный UUID перед запуском стрима
                    session = sessionTransactionService.saveSessionForce(newSession)

                    // ХАК ДЛЯ ДЕМО: Отправляем UUID сессии первым техническим чанком на фронтенд,
                    // чтобы фронт зафиксировал его и присылал в следующих запросах.
                    sink.next("[SESSION_ID:${session.id}]")
                } else {
                    log.info { "🎯 [Оркестратор] Сессия успешно найдена по UUID: ${session.id}. Текущая платформа: ${session.context.platform}. Продолжаю стрим..." }
                }

                // ФИКС №2: Нативно подписываемся на Flux токенов от DynamicAiOrchestrator
                // Нам больше не нужно нарезать текст через split, токены летят прямо из LLM
                dynamicAiOrchestrator.orchestrateDynamicStream(session, userInput)
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
