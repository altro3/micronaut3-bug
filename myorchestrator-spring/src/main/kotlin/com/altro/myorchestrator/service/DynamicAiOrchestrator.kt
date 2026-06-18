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

    @Qualifier("routerAgentClient") private val routerAgentClient: ChatClient,
    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        return Mono.fromCallable {
            log.info { "🏁 [Оркестратор] Начало шага. Текущая платформа в БД: ${session.context.platform ?: "НЕ ОПРЕДЕЛЕНА"}" }

            // --- ШАГ 1: МАРШРУТИЗАЦИЯ СТРОГО ЧЕРЕЗ ENUM PLATFORM ---
            var currentPlatform = session.context.platform
            val isUserTriggeringAutomation = isUserAskingForPlatformSwitch(userInput)

            // Вызываем ИИ-маршрутизатор только один раз на старте сессии,
            // либо если пользователь в процессе свободного CHAT-диалога явно упомянул ключевые слова сетей
            if (currentPlatform == null || (currentPlatform == Platform.CHAT && isUserTriggeringAutomation)) {
                log.info { "🧩 [Роутер] Вызов ИИ-классификатора интентов..." }

                currentPlatform = routeUserInputToPlatform(userInput)
                session.context.platform = currentPlatform

                log.info { "💾 [Роутер] Значение Platform.$currentPlatform зафиксировано в БД сессии. Роутер засыпает." }
            } else {
                log.info { "🎯 [Роутер] СКИП: Платформа '$currentPlatform' уже зафиксирована в БД сессии. Токены сохранены." }
            }

            // --- ШАГ 2: ПОДБОР ПРАВИЛ ИЗ QDRANT НА ОСНОВЕ ENUM ---
            // Для свободного чата (Platform.CHAT) правила ищем в общем контексте (передаем пустой список площадок)
            val filterPlatforms = if (currentPlatform == Platform.CHAT) emptyList() else listOf(currentPlatform)
            val moderationRules = vectorStoreService.searchAdvertisingRules(filterPlatforms, userInput)

            // --- ШАГ 3: ПОДБОР ИЗОЛИРОВАННОГО ИИ-АГЕНТА ---
            val (selectedAgent, agentSystemPrompt) = when (currentPlatform) {
                Platform.YANDEX_DIRECT -> {
                    yandexAgentClient to getYandexSystemPrompt(session.context.campaignId, moderationRules)
                }

                Platform.VK_ADS -> {
                    vkAgentClient to getVkSystemPrompt(moderationRules)
                }

                Platform.CHAT -> {
                    consultantAgentClient to getConsultantSystemPrompt(moderationRules)
                }
            }

            // --- ШАГ 4: СБОРКА ИСТОРИИ ДИАЛОГА ---
            val historyMessages = mutableListOf<Message>().apply {
                add(SystemMessage(agentSystemPrompt))
                session.context.executionLogs.forEachIndexed { index, logText ->
                    if (index % 2 == 0) add(UserMessage(logText)) else add(AssistantMessage(logText))
                }
                add(UserMessage(userInput))
            }

            log.info { "🚀 [Оркестратор] Делегирую управление Агенту [$currentPlatform]. Синхронный Function Calling..." }

            // --- ШАГ 5: СИНХРОННЫЙ ВЫЗОВ ВЫБРАННОГО АГЕНТА С ПОДДЕРЖКОЙ MCP ---
            val chatResponse = selectedAgent.prompt()
                .messages(historyMessages)
                .call()
                .chatResponse()

            val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сгенерировать ответ."

            // --- ШАГ 6: АКТУАЛИЗАЦИЯ СОСТОЯНИЯ И БД СЕССИИ ---
            val currentSession = sessionRepository.findByIdOrNull(session.id) ?: session

            val updatedLogs = currentSession.context.executionLogs.toMutableList().apply {
                add(userInput)
                add(finalResponseText)
            }

            currentSession.context.executionLogs = updatedLogs
            currentSession.updatedAt = Instant.now()

            // Если Консультант в чате выдал триггер перехода к сборке, сбрасываем стейт платформы в null,
            // чтобы на следующем шаге ИИ-роутер автоматически переключил сессию на технического агента
            if (currentPlatform == Platform.CHAT && isTriggerTextToStartAutomation(finalResponseText)) {
                currentSession.context.platform = null
                log.info { "🔄 [Консультант] Зафиксирован текстовый переход к сборке. Сбрасываю стейт для ИИ-роутера." }
            }

            // Извлечение числового ID черновика для инструментов Яндекса
            if (currentPlatform == Platform.YANDEX_DIRECT && currentSession.context.campaignId == null) {
                val extractedId = regexFindId(finalResponseText)
                if (extractedId != null) {
                    currentSession.context.campaignId = extractedId
                    log.info { "🔍 [Яндекс Агент] Успешно сохранен campaignId: $extractedId" }
                }
            }

            sessionRepository.save(currentSession)
            log.info { "💾 [Оркестратор] Шаг успешно завершен. Состояние зафиксировано в репозиторий." }

            finalResponseText
        }
            .subscribeOn(Schedulers.boundedElastic())
            .flatMapMany { text ->
                val tokens = text.split("(?<= )".toRegex())
                Flux.fromIterable(tokens)
            }
    }

    /**
     * Быстрый запрос к ИИ-диспетчеру. Возвращает строго типизированный Platform enum.
     */
    private fun routeUserInputToPlatform(userInput: String): Platform {
        return try {
            val routerPrompt = """
                Проанализируй сообщение пользователя и классифицируй его намерение.
                Выдай в ответ строго одно слово из трех вариантов:
                
                1. YANDEX_DIRECT — если пользователь явно просит настроить, запустить или создать рекламу в Яндекс.Директ.
                2. VK_ADS — если пользователь явно просит настроить, запустить или создать рекламу во ВКонтакте (VK Ads).
                3. CHAT — если пользователь просто здоровается, задает общие вопросы, просит совета, сравнивает площадки или просто общается.
                
                Сообщение пользователя: '$userInput'
                Твой ответ (строго одно слово из трех вариантов):
            """.trimIndent()

            val response = routerAgentClient.prompt()
                .user(routerPrompt)
                .call()
                .content()
                ?.trim()?.uppercase() ?: "CHAT"

            // Превращаем безопасную строку ответа строго в валидный enum тип
            when {
                response.contains("YANDEX") -> Platform.YANDEX_DIRECT
                response.contains("VK") -> Platform.VK_ADS
                else -> Platform.CHAT
            }
        } catch (e: Exception) {
            log.error(e) { "Ошибка ИИ-маршрутизации трафика, фолбек на Platform.CHAT" }
            Platform.CHAT
        }
    }

    private fun isUserAskingForPlatformSwitch(text: String): Boolean {
        val lower = text.lowercase()
        return lower.contains("яндекс") || lower.contains("директ") || lower.contains("вк ") || lower.contains("vk") || lower.contains("вконтакте")
    }

    private fun isTriggerTextToStartAutomation(text: String): Boolean {
        val lower = text.lowercase()
        return lower.contains("приступаю к сборке") || lower.contains("начнем настройку") || lower.contains("переходим к кабинету")
    }

    private fun getYandexSystemPrompt(campaignId: Long?, moderationRules: String): String = """
        Ты — ИИ-Агент Яндекс.Директ. Твоя цель — настроить кампанию в Яндексе с помощью доступных инструментов.
        ${if (campaignId != null) "ТЕКУЩИЙ ID РЕКЛАМНОЙ КАМПАНИИ В ЯНДЕКСЕ: $campaignId." else ""}
        СТРОГИЕ ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:
        1. Извлекай из реплик ГЕО, возраст, имя, тексты и СРАЗУ вызывай инструменты (Slot Filling).
        2. Если нет ID кампании — начни с 'initYandexDraft'.
        3. Если пользователь назвал регион текстом — сначала найди его ID через 'searchYandexRegions', и только затем привяжи через 'bindYandexRegions'. Не гадай ID!
        4. Когда готов текст объявления — вызывай 'submitYandexCreative'.
        5. Публикуй ('publishYandexCampaign') строго после явного согласия.
        ЗАПРЕТ НА АНОНСЫ: Никогда не пиши "Сейчас я вызову инструмент...". Сначала молча вызывай инструмент.
        🎯 Правила Директа: $moderationRules
    """.trimIndent()

    private fun getVkSystemPrompt(moderationRules: String): String = """
        Ты — ИИ-Агент VK Ads. Твоя цель — создавать рекламные кампании на платформе ВКонтакте.
        Используй инструмент 'createVkCampaign' для инициации рекламного процесса, когда пользователь определился с именем кампании, заголовком и текстом баннера.
        ЗАПРЕТ НА АНОНСЫ: Никогда не пиши "Сейчас я вызову инструмент...". Сначала молча вызывай инструмент.
        🎯 Правила VK Ads: $moderationRules
    """.trimIndent()

    private fun getConsultantSystemPrompt(moderationRules: String): String = """
        Ты — опытный и дружелюбный ИИ-маркетолог рекламного агрегатора AdBroker.
        Твоя цель — вести свободный диалог, отвечать на вопросы про маркетинг, помогать придумывать стратегии, креативы и анализировать ниши.
        ПРАВИЛА ПОВЕДЕНИЯ:
        1. Если пользователь просто приветствует тебя или общается на свободные темы — поддерживай диалог как высококлассный эксперт.
        2. Если пользователь сомневается, какую сеть выбрать, объясни разницу: Яндекс.Директ хорош для горячего спроса, а VK Ads — для прогрева через сообщества и таргетинг по интересам.
        3. Как только пользователь четко скажет, что готов выбрать конкретную сеть, вежливо зафиксируй это в ответе и напиши СТРОГО одну из фраз-триггеров: "Приступаю к сборке кампании в Яндекс.Директ" или "Приступаю к сборке кампании в VK Ads".
        🎯 Полезная общая информация из базы знаний: $moderationRules
    """.trimIndent()

    private fun regexFindId(text: String): Long? {
        val regex = "\"id\"\\s*:\\s*(\\d+)".toRegex()
        val matchResult = regex.find(text)
        return matchResult?.groupValues?.get(1)?.toLongOrNull()
    }
}