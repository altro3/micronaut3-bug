package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpAsyncClient
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.mcp.AsyncMcpToolCallbackProvider
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers
import java.util.concurrent.Executors

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val mcpClient: McpAsyncClient,
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStoreService: VectorStoreService,
) {

    private val mcpScheduler = Schedulers.fromExecutor(Executors.newFixedThreadPool(4))

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        session.currentStep = CampaignCreationStep.AI_DYNAMIC_COLLECTING

        val currentPlatforms = listOfNotNull(session.context.selectedPlatform)

        val moderationRules = vectorStoreService.searchAdvertisingRules(
            platforms = currentPlatforms,
            userBriefText = userInput,
        )

        val systemPrompt = """
            Ты — ИИ-оркестратор рекламного агрегатора. Твоя цель — помочь пользователю настроить кампанию в Яндекс.Директ или VK Ads, используя доступные тебе MCP-инструменты.
            Ты ведешь живой диалог, извлекаешь параметры и управляешь состоянием бэкенда.
            
            СТРОГИЕ ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:
            1. Платформа: Если пользователь не указал, где именно он хочет запустить рекламу (Яндекс или VK), ты ОБЯЗАН сначала вежливо уточнить у него целевую платформу.
            2. Свободный ввод (Slot Filling): Извлекай из реплик ГЕО, возраст, имя, тексты и СРАЗУ вызывай соответствующие тулы. Не задавай пошаговых вопросов на то, что пользователь уже назвал.
            3. Инициация: Если у тебя в контексте нет ID кампании (campaignId), ты ОБЯЗАН начать с вызова 'initYandexDraft' (для Яндекса) или соответствующего тула для VK. Полученный ID используй во всех остальных тулах.
            4. Работа с ГЕО: Если пользователь назвал регион текстом (например, 'Москва'), ты НЕ ИМЕЕШЬ ПРАВА гадать ID. Сначала вызови 'searchYandexRegions', найди точный ID из ответа бэкенда, и только потом вызови 'bindYandexRegions'.
            
            КРИТИЧЕСКОЕ ПРАВИЛО РАЗДЕЛЕНИЯ КРЕАТИВА И ПУБЛИКАЦИИ:
            5. Сохранение креатива: Когда пользователь дает текст объявления или просит его придумать, вызывай ИНСТРУМЕНТ 'submitYandexCreative'. Это ТОЛЬКО сохраняет текст в черновик.
               - После этого покажи текст пользователю и спроси явное согласие на запуск (например: "Я сохранил текст. Запускаем рекламу в сеть?").
            6. Финальная публикация: Вызывай инструмент 'publishYandexCampaign' СТРОГО тогда, когда текст уже сохранен, ВСЕ поля заполнены, и пользователь дал прямое согласие (написал 'Да', 'Запускай', 'Публикуй', 'Ок'). Никогда не публикуй кампанию в сеть автоматически без команды подтверждения!
            
            Если какой-то инструмент возвращает ошибку (например, ошибка валидации Яндекса в JSON), прочитай её, объясни пользователю человеческим языком и предложи исправление.
            
            Правила сетей и лимиты символов для генерации (с учетом ниши бизнеса):
            $moderationRules
        """.trimIndent()

        val historyMessages = mutableListOf<Message>().apply {
            add(SystemMessage(systemPrompt))
            session.context.executionLogs.forEachIndexed { index, log ->
                if (index % 2 == 0) add(UserMessage(log)) else add(AssistantMessage(log))
            }
            add(UserMessage(userInput))
        }

        val toolProvider = AsyncMcpToolCallbackProvider.builder()
            .mcpClients(listOf(mcpClient))
            .build()

        val stringBuffer = StringBuilder()

        return chatClientBuilder.build()
            .prompt()
            .messages(historyMessages)
            .tools(toolProvider)
            .stream()
            .content()
            .doOnNext { stringBuffer.append(it) }
            .doOnComplete {
                val updatedLogs = session.context.executionLogs.toMutableList().apply {
                    add(userInput)
                    add(stringBuffer.toString())
                }
                val extractedId = regexFindId(stringBuffer.toString())
                if (extractedId != null) {
                    log.info { "🎯 [ИИ-Оркестратор] Нашел сквозной ID кампании Яндекса: $extractedId. Фиксирую в сессию." }
                    session.context.yadCampaignId = extractedId
                }
                session.context.executionLogs = updatedLogs
                sessionRepository.save(session)
            }
            .subscribeOn(mcpScheduler)
    }

    private fun regexFindId(text: String): Long? {
        val regex = "\"id\"\\s*:\\s*(\\d+)".toRegex()
        val matchResult = regex.find(text)
        return matchResult?.groupValues?.get(1)?.toLongOrNull()
    }
}
