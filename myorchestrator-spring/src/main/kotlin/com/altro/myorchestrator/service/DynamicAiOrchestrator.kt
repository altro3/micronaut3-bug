package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono
import reactor.core.scheduler.Schedulers
import java.time.Instant

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val chatClient: ChatClient,
    private val vectorStoreService: VectorStoreService,
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        return Mono.fromCallable {
            val currentPlatforms = listOfNotNull(session.context.platform)
            val currentCampaignId = session.context.campaignId
            val moderationRules = vectorStoreService.searchAdvertisingRules(currentPlatforms, userInput)

            val systemPrompt = """
                Ты — ИИ-оркестратор рекламного агрегатора. Твоя цель — помочь пользователю настроить кампанию в Яндекс.Директ, используя доступные тебе инструменты.
                Ты ведешь живой диалог, извлекаешь параметры и управляешь состоянием бэкенда.
                
                СТРОГИЕ ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:
                1. Свободный ввод (Slot Filling): Извлекай из реплик ГЕО, возраст, имя, тексты и СРАЗУ вызывай соответствующие инструменты. Не задавай пошаговых вопросов на то, что пользователь уже назвал.
                2. Инициация: Если у тебя в контексте нет ID кампании (campaignId), ты ОБЯЗАН начать с вызова 'initYandexDraft'. Полученный ID используй во всех остальных инструментах.
                3. Работа с ГЕО: Если пользователь назвал регион текстом (например, 'Москва'), ты НЕ ИМЕЕШЬ ПРАВА гадать ID. Сначала вызови 'searchYandexRegions', найди точный ID из ответа бэкенда, и только потом вызови 'bindYandexRegions'.
                4. Сохранение креатива: Когда пользователь дает текст объявления или просит его придумать, вызывай 'submitYandexCreative'. Это ТОЛЬКО сохраняет текст в черновик. После этого спроси согласие на запуск.
                5. Финальная публикация: Вызывай инструмент 'publishYandexCampaign' СТРОГО тогда, когда текст уже сохранен и пользователь дал прямое согласие ('Да', 'Запускай', 'Публикуй').
                6. ЗАПРЕТ НА АНОНСЫ ДЕЙСТВИЙ: Никогда не пиши пользователю текст вида "Сейчас я вызову инструмент...", "Я собираюсь найти ID..." или "Сейчас я привяжу регион...". 
                    Ты НЕ должен комментировать свои намерения. Ты ОБЯЗАН сразу молча вызывать соответствующий инструмент (`searchYandexRegions`, `bindYandexRegions` и т.д.). 
                    Человеческий текст пользователю ты пишешь ТОЛЬКО ТОГДА, когда бэкенд уже вернул тебе реальный результат выполнения инструмента!

                ${if (currentCampaignId != null) "ТЕКУЩИЙ ID КАМПАНИИ, С КОТОРОЙ ТЫ РАБОТАЕШЬ: $currentCampaignId. Используй строго его во всех инструментах Яндекса!" else ""}
                
                ПРАВИЛО ПОВЕДЕНИЯ ПОСЛЕ ОПЕРАЦИИ:
                Как только ты успешно выполнил инструмент 'initYandexDraft' и бэкенд вернул тебе ID, прекрати вызовы функций в этом шаге и напиши пользователю человеческий текст. 
                Сообщи, что кампания создана, и вежливо спроси, в каких регионах или городах запускать рекламу. Ни в коем случае не присылай пустой ответ!
                
                🎯 Правила сетей и лимиты символов для генерации, которые ты обязан соблюдать:
                $moderationRules
            """.trimIndent()

            // Безопасное восстановление истории диалога
            val historyMessages = mutableListOf<Message>().apply {
                add(SystemMessage(systemPrompt))
                session.context.executionLogs.forEachIndexed { index, logText ->
                    if (index % 2 == 0) add(UserMessage(logText)) else add(AssistantMessage(logText))
                }
                add(UserMessage(userInput))
            }

            log.info { "🚀 [DynamicOrchestrator] Отправка запроса в Qwen с поддержкой сетевого MCP..." }

            val chatResponse = chatClient.prompt()
                .messages(historyMessages)
                .call()
                .chatResponse()

            val finalResponseText = chatResponse?.result?.output?.text ?: "Модель не смогла сгенерировать ответ."
            val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session

            val updatedLogs = currentSession.context.executionLogs.toMutableList().apply {
                add(userInput)
                add(finalResponseText)
            }

            currentSession.context.executionLogs = updatedLogs
            currentSession.updatedAt = Instant.now()

            if (currentSession.context.campaignId == null) {
                val extractedId = regexFindId(finalResponseText)
                if (extractedId != null) {
                    currentSession.context.campaignId = extractedId
                    log.info { "🔍 ID кампании ($extractedId) вытащен регуляркой из текста ответа" }
                }
            }

            sessionRepository.save(currentSession)
            finalResponseText
        }
            .subscribeOn(Schedulers.boundedElastic())
            .flatMapMany { text ->
                val tokens = text.split("(?<= )".toRegex())
                Flux.fromIterable(tokens)
            }
    }

    private fun regexFindId(text: String): Long? {
        val regex = "\"id\"\\s*:\\s*(\\d+)".toRegex()
        val matchResult = regex.find(text)
        return matchResult?.groupValues?.get(1)?.toLongOrNull()
    }
}
