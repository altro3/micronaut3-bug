package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono
import reactor.core.scheduler.Schedulers
import java.time.Instant

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val vectorStoreService: VectorStoreService,
    @Qualifier("routerAgentClient")
    private val routerAgentClient: ChatClient,
    @Qualifier("yandexAgentClient")
    private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient")
    private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient")
    private val consultantAgentClient: ChatClient,
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        return Mono.fromCallable {
            log.info { "🏁 [Оркестратор] Начало обработки шага. Текущая платформа в сессии: ${session.context.platform ?: "НЕ ЗАДАНА"}" }

            // --- ШАГ 1: ОПРЕДЕЛЕНИЕ ИНТЕНТА / МАРШРУТИЗАЦИЯ ---
            var platform = session.context.platform

            if (platform == null) {
                log.info { "🧩 [Роутер] Платформа не зафиксирована. Запрашиваю классификацию намерения..." }
                val routingIntent = routeUserInputToIntent(userInput)
                log.info { "🧩 [Роутер] Результат классификации интента: '$routingIntent'" }

                if (routingIntent == "YANDEX" || routingIntent == "VK") {
                    platform = Platform.valueOf(if (routingIntent == "YANDEX") "YANDEX_DIRECT" else "VK_ADS")
                    session.context.platform = platform
                    log.info { "🎯 [Роутер] Сессия жестко переведена в режим настройки платформы: $platform" }
                }
            }

            // --- ШАГ 2: ПОДБОР АГЕНТА И ОБОГАЩЕНИЕ КОНТЕКСТА ИЗ QDRANT ---
            val currentPlatforms = listOfNotNull(platform)
            val moderationRules = vectorStoreService.searchAdvertisingRules(currentPlatforms, userInput)

            val (selectedAgent, agentSystemPrompt) = when {
                platform == Platform.YANDEX_DIRECT -> {
                    yandexAgentClient to getYandexSystemPrompt(session.context.campaignId, moderationRules)
                }

                platform == Platform.VK_ADS -> {
                    vkAgentClient to getVkSystemPrompt(moderationRules)
                }

                else -> {
                    // Если платформа не выбрана (свободный диалог) — передаем управление Консультанту
                    consultantAgentClient to getConsultantSystemPrompt(moderationRules)
                }
            }

            // --- ШАГ 3: БЕЗОПАСНОЕ ВОССТАНОВЛЕНИЕ ИСТОРИИ ДИАЛОГА ---
            val historyMessages = mutableListOf<Message>().apply {
                add(SystemMessage(agentSystemPrompt))
                session.context.executionLogs.forEachIndexed { index, logText ->
                    if (index % 2 == 0) add(UserMessage(logText)) else add(AssistantMessage(logText))
                }
                add(UserMessage(userInput))
            }

            log.info { "🚀 [Оркестратор] Запуск выполнения выбранного агента. Синхронный Function Calling..." }

            // --- ШАГ 4: СИНХРОННЫЙ ВЫЗОВ АГЕНТА С ПОДДЕРЖКОЙ MCP ---
            val chatResponse = selectedAgent.prompt()
                .messages(historyMessages)
                .call()
                .chatResponse()

            val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сгенерировать ответ."

            // --- ШАГ 5: АКТУАЛИЗАЦИЯ СОСТОЯНИЯ СЕССИИ И БД ---
            val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session

            val updatedLogs = currentSession.context.executionLogs.toMutableList().apply {
                add(userInput)
                add(finalResponseText)
            }

            currentSession.context.executionLogs = updatedLogs
            currentSession.updatedAt = Instant.now()

            // Если работал Яндекс-агент и в базе сессии еще нет ID кампании — парсим и сохраняем его
            if (platform == Platform.YANDEX_DIRECT && currentSession.context.campaignId == null) {
                val extractedId = regexFindId(finalResponseText)
                if (extractedId != null) {
                    currentSession.context.campaignId = extractedId
                    log.info { "🔍 [Яндекс Агент] Успешно вытащили и сохранили campaignId: $extractedId" }
                }
            }

            sessionRepository.save(currentSession)
            log.info { "💾 [Оркестратор] Состояние сессии сохранено. Шаг успешно завершен." }

            finalResponseText
        }
            .subscribeOn(Schedulers.boundedElastic())
            // Нарезаем финальный ответ по пробелам для эмуляции SSE-стриминга (живой чат на фронтенде)
            .flatMapMany { text ->
                val tokens = text.split("(?<= )".toRegex())
                Flux.fromIterable(tokens)
            }
    }

    /**
     * Запрос к модели для быстрого определения намерения (ИИ-Диспетчер)
     */
    private fun routeUserInputToIntent(userInput: String): String {
        return try {
            // Формируем строгий промпт для классификации, чтобы Qwen не умничала
            val routerPrompt = """
                Проанализируй сообщение пользователя и классифицируй его намерение.
                Выдай в ответ строго одно слово из трех вариантов:
                
                1. YANDEX — если пользователь явно просит настроить, запустить или создать рекламу в Яндекс.Директ.
                2. VK — если пользователь явно просит настроить, запустить или создать рекламу во ВКонтакте (VK Ads).
                3. CHAT — если пользователь просто здоровается, задает общие вопросы, просит совета, сравнивает площадки или просто общается.
                
                Сообщение пользователя: '$userInput'
                Твой ответ (строго одно слово):
            """.trimIndent()

            routerAgentClient.prompt()
                .user(routerPrompt)
                .call()
                .content()
                ?.trim()?.uppercase() ?: "CHAT"
        } catch (e: Exception) {
            log.error(e) { "Ошибка ИИ-маршрутизации трафика, фолбек на свободный чат" }
            "CHAT"
        }
    }

    /**
     * Системные инструкции для Агента Яндекс.Директ
     */
    private fun getYandexSystemPrompt(campaignId: Long?, moderationRules: String): String = """
        Ты — ИИ-Агент Яндекс.Директ. Твоя единственная цель — настроить кампанию в Яндексе с помощью доступных инструментов.
        Ты ведешь живой диалог, извлекаешь параметры и управляешь состоянием бэкенда.
        
        ${if (campaignId != null) "ТЕКУЩИЙ ID РЕКЛАМНОЙ КАМПАНИИ В ЯНДЕКСЕ: $campaignId. Используй строго его во всех инструментах!" else ""}
        
        СТРОГИЕ ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:
        1. Свободный ввод (Slot Filling): Извлекай из реплик ГЕО, возраст, имя, тексты и СРАЗУ вызывай соответствующие инструменты. Не задавай пошаговых вопросов на то, что пользователь уже назвал.
        2. Инициация: Если у тебя в контексте нет ID кампании (campaignId), ты ОБЯЗАН начать с вызова 'initYandexDraft'. Полученный ID используй во всех остальных инструментах.
        3. Работа с ГЕО: Если пользователь назвал регион текстом (например, 'Москва'), ты НЕ ИМЕЕШЬ ПРАВА гадать ID. Сначала вызови 'searchYandexRegions', найди точный ID из ответа бэкенда, и только потом вызови 'bindYandexRegions'.
        4. Сохранение креатива: Когда пользователь дает текст объявления или просит его придумать, вызывай 'submitYandexCreative'. Это ТОЛЬКО сохраняет текст в черновик. После этого спроси согласие на запуск.
        5. Финальная публикация: Вызывай инструмент 'publishYandexCampaign' СТРОГО тогда, когда текст уже сохранен и пользователь дал прямое согласие ('Да', 'Запускай', 'Публикуй').
        
        ПРАВИЛО ПОВЕДЕНИЯ ПОСЛЕ ОПЕРАЦИИ (ЗАПРЕТ НА АНОНСЫ):
        Никогда не пиши пользователю текст вида "Сейчас я вызову инструмент...", "Я собираюсь найти ID..." или "Сейчас я привяжу регион...". Ты НЕ должен комментировать свои намерения. Ты ОБЯЗАН сразу молча вызывать соответствующий инструмент.
        Человеческий текст пользователю ты пишешь ТОЛЬКО ТОГДА, когда бэкенд уже вернул тебе реальный результат выполнения инструмента! Как только ты успешно выполнил инструмент 'initYandexDraft' и бэкенд вернул тебе ID, прекрати вызовы функций в этом шаге и напиши пользователю человеческий текст: сообщи, что кампания создана, и вежливо спроси, в каких регионах или городах запускать рекламу.
        
        🎯 Правила сети Яндекс.Директ и лимиты символов, которые ты обязан соблюдать:
        $moderationRules
    """.trimIndent()

    /**
     * Системные инструкции для Агента ВКонтакте
     */
    private fun getVkSystemPrompt(moderationRules: String): String = """
        Ты — ИИ-Агент VK Ads. Твоя цель — создавать рекламные кампании на платформе ВКонтакте.
        Используй доступный инструмент 'createVkCampaign' для инициации рекламного процесса, когда пользователь определился с именем кампании, заголовком и текстом баннера.
        
        ЗАПРЕТ НА АНОНСЫ: Никогда не пиши "Сейчас я вызову инструмент...". Сначала молча вызывай инструмент. Человеческий текст пиши строго по результатам ответа от бэкенда.
        
        🎯 Правила сети VK Ads и лимиты символов для генерации, которые ты обязан соблюдать:
        $moderationRules
    """.trimIndent()

    /**
     * Системные инструкции для Агента-Консультанта (Свободный диалог)
     */
    private fun getConsultantSystemPrompt(moderationRules: String): String = """
        Ты — опытный, дружелюбный и проактивный ИИ-маркетолог рекламного агрегатора AdBroker.
        Твоя цель — вести свободный диалог с пользователем, отвечать на его вопросы про маркетинг, помогать придумывать стратегии, креативы и анализировать ниши.
        
        ПРАВИЛА ПОВЕДЕНИЯ:
        1. Если пользователь просто приветствует тебя или общается на свободные темы — поддерживай диалог как высококлассный эксперт.
        2. Если пользователь сомневается, какую сеть выбрать, объясни разницу: Яндекс.Директ идеален для горячего спроса (когда услугу ищут прямо сейчас в поиске), а VK Ads — для прогрева аудитории через сообщества, таргетинг по интересам и яркие баннеры.
        3. Как только в процессе разговора пользователь четко скажет: "Давай настраивать Яндекс" или "Хочу запустить кампанию в ВК", вежливо зафиксируй это в ответе и скажи, что ты приступаешь к сборке.
        
        🎯 Полезная общая информация по рекламным правилам из базы знаний:
        $moderationRules
    """.trimIndent()

    private fun regexFindId(text: String): Long? {
        val regex = "\"id\"\\s*:\\s*(\\d+)".toRegex()
        val matchResult = regex.find(text)
        return matchResult?.groupValues?.get(1)?.toLongOrNull()
    }
}
