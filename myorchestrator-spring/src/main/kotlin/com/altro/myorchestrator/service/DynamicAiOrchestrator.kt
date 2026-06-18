package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import tools.jackson.databind.json.JsonMapper
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap

@Service
class DynamicAiOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val vectorStoreService: VectorStoreService,
    private val jsonMapper: JsonMapper,
    private val sessionTransactionService: SessionTransactionService,
    @Qualifier("routerAgentClient") private val routerAgentClient: ChatClient,
    @Qualifier("yandexAgentClient") private val yandexAgentClient: ChatClient,
    @Qualifier("vkAgentClient") private val vkAgentClient: ChatClient,
    @Qualifier("consultantAgentClient") private val consultantAgentClient: ChatClient
) {

    private val log = KotlinLogging.logger {}

    fun orchestrateDynamicStream(session: CampaignSession, userInput: String): Flux<String> {
        val responseAccumulator = StringBuilder()

        // --- ШАГ 1: МГНОВЕННАЯ СЕМАНТИЧЕСКАЯ МАРШРУТИЗАЦИЯ И ДЕЛЕГАЦИЯ ---
        var currentPlatform = session.context.platform
        val isUserForcingSwitch = isUserAskingForPlatformSwitch(userInput)

        log.info { "🧩 [Входной шаг] Текущий статус в БД: $currentPlatform, триггер смены: $isUserForcingSwitch" }

        if (currentPlatform == null || (currentPlatform == Platform.CHAT && isUserForcingSwitch)) {
            log.info { "🧩 [Роутер] Анализирую намерение пользователя на лету через ИИ-классификатор..." }
            val newPlatform = routeUserInputToPlatform(userInput)

            if (newPlatform != currentPlatform) {
                log.info { "🔄 [МГНОВЕННАЯ ДЕЛЕГАЦИЯ] Переключаю управление с $currentPlatform на целевого агента $newPlatform" }
                currentPlatform = newPlatform

                // Мутируем поля обычного класса напрямую
                session.context.platform = currentPlatform
                if (currentPlatform != Platform.CHAT) {
                    session.context.campaignId = null // Сбрасываем ID для новой автоматизации Яндекса/ВК
                }
            }
        }

        // --- ШАГ 2: ПОДБОР ПРАВИЛ ИЗ QDRANT НА ОСНОВЕ ЦЕЛЕВОЙ ПЛАТФОРМЫ ---
        val filterPlatforms = if (currentPlatform == Platform.CHAT) emptyList() else listOf(currentPlatform)
        val moderationRules = vectorStoreService.searchAdvertisingRules(filterPlatforms, userInput)

        // --- ШАГ 3: ПОДБОР ЦЕЛЕВОГО ИЗОЛИРОВАННОГО ИИ-АГЕНТА ---
        val (selectedAgent, agentSystemPrompt) = when (currentPlatform) {
            Platform.YANDEX_DIRECT -> yandexAgentClient to getYandexSystemPrompt(session.context.campaignId, moderationRules)
            Platform.VK_ADS -> vkAgentClient to getVkSystemPrompt(moderationRules)
            Platform.CHAT -> consultantAgentClient to getConsultantSystemPrompt(moderationRules)
        }

        // --- ШАГ 4: ПОЛНОЕ ВОССТАНОВЛЕНИЕ ТЕХНИЧЕСКОЙ ИСТОРИИ ДЛЯ ВЫБРАННОГО АГЕНТА ---
        val historyMessages = mutableListOf<Message>().apply {
            add(SystemMessage(agentSystemPrompt)) // Свежий системный промпт со всеми лимитами и ID встает первым

            session.context.executionLogs.forEach { jsonStr ->
                try {
                    add(jsonMapper.readValue(jsonStr, Message::class.java))
                } catch (e: Exception) {
                    log.warn { "Ошибка десериализации технического лога истории: ${e.message}" }
                }
            }
        }

        val tempMemory = MessageWindowChatMemory.builder()
            .chatMemoryRepository(InMemoryChatMemoryRepository())
            .build()
        val conversationId = session.id.toString()
        tempMemory.add(conversationId, historyMessages)

        val startCount = historyMessages.size

        // --- ШАГ 5: ВЫПОЛНЕНИЕ В ЗАВИСИМОСТИ ОТ ТИПА АГЕНТА (ЗАЩИТА ТРАНЗАКЦИЙ) ---
        if (currentPlatform == Platform.CHAT) {
            log.info { "💬 [Стрим] Запуск ИИ-Консультанта в реактивном режиме..." }

            return selectedAgent.prompt()
                .user(userInput)
                .advisors(MessageChatMemoryAdvisor.builder(tempMemory).build())
                .advisors { it.param("chat_memory_conversation_id", conversationId) }
                .stream()
                .content()
                .doOnNext { token -> responseAccumulator.append(token) }
                .doOnTerminate {
                    try {
                        val finalResponseText = responseAccumulator.toString()
                        val pagedMessages = tempMemory.get(conversationId)
                        val newTechnicalMessages = if (pagedMessages.size > startCount) pagedMessages.subList(startCount, pagedMessages.size) else emptyList()

                        val currentSession = sessionRepository.findByIdOrNull(session.id ?: error("ID is null")) ?: session
                        val serializedNewLogs = serializeMessagesToLogs(newTechnicalMessages)

                        val nextPlatform = if (isTriggerTextToStartAutomation(finalResponseText)) {
                            if (finalResponseText.contains("Яндекс", ignoreCase = true)) Platform.YANDEX_DIRECT else Platform.VK_ADS
                        } else {
                            Platform.CHAT
                        }

                        // Прямая классическая мутация полей
                        currentSession.context.executionLogs = currentSession.context.executionLogs + serializedNewLogs
                        currentSession.context.platform = nextPlatform
                        currentSession.updatedAt = Instant.now()

                        sessionTransactionService.saveSessionForce(currentSession)
                        log.info { "💾 [БАЗА ЧАТА] Шаг свободной беседы зафиксирован. Следующий статус: $nextPlatform" }
                    } catch (e: Exception) {
                        log.error(e) { "Ошибка сохранения логов чата" }
                    }
                }
        } else {
            // ТЯЖЕЛЫЕ АГЕНТЫ АВТОМАТИЗАЦИИ (Яндекс/ВК): Выполняем синхронно в Tomcat-потоке!
            log.info { "🚀 [Оркестратор] Вызов тяжелого ИИ-Агента автоматизации [$currentPlatform] с MCP инструментами..." }

            var clientPrompt = selectedAgent.prompt()
                .user(userInput)
                .advisors(MessageChatMemoryAdvisor.builder(tempMemory).build())
                .advisors { it.param("chat_memory_conversation_id", conversationId) }

            // Если кампания уже создана, вешаем жесткое системное напоминание в самый хвост перед ответом
            if (session.context.campaignId != null) {
                log.info { "🔒 [ЗАЩИТА ОТ ДУБЛИКАТОВ] Внедряю принудительный запрет на повторный initYandexDraft для ID: ${session.context.campaignId}" }
                clientPrompt = clientPrompt.system("""
                    ВНИМАНИЕ! Рекламная кампания УЖЕ успешно создана. 
                    Текущий числовой ID кампании в системе равен строго: ${session.context.campaignId}.
                    Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО вызывать инструмент 'initYandexDraft' заново!
                    Используй исключительно готовый ID ${session.context.campaignId} во всех остальных инструментах привязки регионов, возраста и креативов.
                """.trimIndent())
            }

            val chatResponse = clientPrompt.call().chatResponse()
            val finalResponseText = chatResponse?.result?.output?.text ?: "Агент не смог сформировать ответ."

            // Вытаскиваем технический след вызовов инструментов из памяти
            val pagedMessages = tempMemory.get(conversationId)
            val newTechnicalMessages = if (pagedMessages.size > startCount) pagedMessages.subList(startCount, pagedMessages.size) else emptyList()

            // Вычитываем текущую сессию из БД по первичному ключу перед коммитом
            val currentSession = sessionRepository.findByIdOrNull(session.id ?: error("ID сессии равен null")) ?: session

            // Извлекаем campaignId из сырых ответов инструментов
            val freshlyExtractedId = extractCampaignIdFromToolResponses(newTechnicalMessages)
            val finalCampaignId = freshlyExtractedId ?: currentSession.context.campaignId

            // Обычная мутация полей объекта
            val serializedNewLogs = serializeMessagesToLogs(newTechnicalMessages)
            currentSession.context.executionLogs += serializedNewLogs
            currentSession.context.campaignId = finalCampaignId
            currentSession.context.platform = currentPlatform
            currentSession.updatedAt = Instant.now()

            log.info { "⏳ [Оркестратор] Отправляю мутированную сущность на принудительный SQL UPDATE в Postgres..." }

            // Физически пишем SQL UPDATE и коммитим транзакцию JDBC прямо сейчас
            val saved = sessionTransactionService.saveSessionForce(currentSession)

            // Синхронизируем входную сессию рантайма, чтобы промпты на лету видели апдейты
            session.context.campaignId = saved.context.campaignId
            session.context.platform = saved.context.platform
            session.context.executionLogs = saved.context.executionLogs

            log.info { "💾 [БАЗА ДАННЫХ ТОМКАТ] Жесткий UPDATE выполнен! ИИ-роль в БД: ${saved.context.platform}, Числовой campaignId в строке: ${saved.context.campaignId}, Общее число логов: ${saved.context.executionLogs.size}" }

            return Flux.just(finalResponseText)
        }
    }

    private fun serializeMessagesToLogs(messages: List<Message>): List<String> {
        return messages.map { msg ->
            val dto = when (msg) {
                is ToolResponseMessage -> {
                    val resp = msg.responses.firstOrNull()
                    val cleanContent = if (resp?.responseData != null) jsonMapper.writeValueAsString(resp.responseData) else msg.text ?: ""
                    PersistentMessageDto("TOOL", cleanContent, resp?.id, resp?.name)
                }

                else -> PersistentMessageDto(msg.messageType.name, msg.text ?: "")
            }
            jsonMapper.writeValueAsString(dto)
        }
    }

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

    private fun extractCampaignIdFromToolResponses(messages: List<Message>): Long? {
        for (msg in messages) {
            if (msg is ToolResponseMessage) {
                try {
                    val resp = msg.responses.firstOrNull()

                    // 1. Пытаемся взять сырой текст из ответа инструмента
                    var rawJson = msg.text ?: ""

                    // 2. Если текста нет, но есть объект responseData, сериализуем его
                    if (rawJson.isBlank() && resp?.responseData != null) {
                        rawJson = jsonMapper.writeValueAsString(resp.responseData)
                    }

                    // Если оба поля пусты — пропускаем это сообщение
                    if (rawJson.isBlank()) continue

                    // Чистим экранированные кавычки, если Jackson обернул строку дважды
                    if (rawJson.startsWith("\"") && rawJson.endsWith("\"") && rawJson.length > 2) {
                        // Распаковываем экранированный JSON, если он прилетел как текстовая строка
                        try {
                            rawJson = jsonMapper.readValue(rawJson, String::class.java)
                        } catch (e: Exception) {
                            // Игнорируем, если это была обычная строка
                        }
                    }

                    val rootNode = jsonMapper.readTree(rawJson)

                    // Ищем строго поле "id", которое возвращает ваш CampaignStepRs DTO!
                    val idNode = rootNode.get("id") ?: rootNode.get("campaignId") ?: rootNode.get("campaign_id")

                    if (idNode != null && idNode.isNumber) {
                        val extractedId = idNode.asLong()
                        log.info { "🎯 [MCP PARSER ЖЕЛЕЗОБЕТОННО] Из логов инструмента '${resp?.name}' успешно извлечен ID кампании: $extractedId" }
                        return extractedId
                    }
                } catch (e: Exception) {
                    log.warn { "Не удалось пропарсить JSON-след инструмента для сбора ID: ${e.message}" }
                }
            }
        }
        return null
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
    
    ${if (campaignId != null) "ТЕКУЩИЙ ID РЕКЛАМНОЙ КАМПАНИИ: $campaignId." else "У тебя еще нет ID кампании. Начни строго с 'initYandexDraft'."}
    
    🚨🚨 ПРАВИЛО ЧЕСТНОСТИ (ЗАПРЕТ НА ЛОЖЬ):
    Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать в текстовом ответе пользователю, что ты выполнил привязку (возраста, региона, ИНН или текстов), ЕСЛИ ТЫ НЕ ВЫЗВАЛ СООТВЕТСТВУЮЩИЙ ИНСТРУМЕНТ в этом шаге! 
    Сначала вызывай инструмент, получай статус "Успешно" от бэкенда, и только потом подтверждай это человеку.
    
    СТРОГИЙ АЛГОРИТМ:
    1. Если ID нет — вызови 'initYandexDraft'. Получи ID.
    2. Если ID есть, но пользователь передал новые параметры (например, ИНН, возраст "18+", или гео):
       - Для возраста — СРАЗУ вызывай 'bindYandexAges'.
       - Для региона — СРАЗУ вызывай 'bindYandexRegions'.
       - Для текстов — СРАЗУ вызывай 'submitYandexCreative'.
    3. Только после того, как ВСЕ доступные параметры переданы в инструменты, спроси у пользователя недостающие (например, радиус и бюджет, как на скриншоте).
    
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
}

data class PersistentMessageDto(
    val messageType: String,
    val content: String,
    val id: String? = null,
    val name: String? = null
)
