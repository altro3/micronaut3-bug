package com.altro.myorchestrator.config

import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CampaignIdInterceptingToolCallback
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor
import org.springframework.ai.chat.memory.ChatMemoryRepository
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.chat.prompt.PromptTemplate
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import tools.jackson.databind.json.JsonMapper

@Configuration
class AiConfig {

    @Bean
    fun consultantAgentClient(
        chatClientBuilder: ChatClient.Builder,
        chatMemoryRepository: ChatMemoryRepository,
        vectorStore: VectorStore,
    ): ChatClient {

        val consultantRagTemplate = PromptTemplate(
            """
        ⚠️ ОБЩИЕ ЗАКОНОДАТЕЛЬНЫЕ ПРАВИЛА И ОГРАНИЧЕНИЯ (ИЗ КВАДРАНТА):

        ---------------------
        {question_answer_context}
        ---------------------

        Запрос пользователя, на который нужно ответить: {query}
        """.trimIndent()
        )

        return chatClientBuilder
            .defaultSystem(
                """
        Ты — опытный и дружелюбный ИИ-маркетолог рекламного агрегатора AdBroker.
        Твоя цель — вести свободный диалог, отвечать на вопросы про маркетинг, помогать придумывать стратегии, креативы и анализировать ниши.
        
        🎯 ТЕХНИЧЕСКИЙ КОНТЕКСТ СЕССИИ:
        {technicalSessionContext}
        
        ПРАВИЛА ПОВЕДЕНИЯ:
        1. Если пользователь просто приветствует тебя или общается на свободные темы — поддерживай диалог как высококлассный эксперт.
        2. Если пользователь сомневается, какую сеть выбрать, объясни разницу.
        3. Как только пользователь четко скажет, что готов выбрать конкретную сеть, вежливо зафиксируй это в ответе и напиши СТРОГО одну из фраз-триггеров: "Приступаю к сборке кампании в Яндекс.Директ" или "Приступаю к сборке кампании в VK Ads".
        
        ПРАВИЛА ОФОРМЛЕНИЯ MARKDOWN РАЗМЕТКИ:
        1. Если ты хочешь визуально отделить один блок текста от другого горизонтальной линией, ты ОБЯЗАН оставлять пустые строки до и после дефисов. 
           Пиши СТРОГО так:
           Текст абзаца.
           
           ---
           
           ### Следующий заголовок
           Если ты не оставишь пустую строку между текстом и '---', твой текст ошибочно превратится в заголовок!
        2. Для оформления списков и таблиц всегда разделяй их с обычным текстом пустой строкой.
        """.trimIndent()
            )
            .defaultAdvisors(
                chatMemoryAdvisor(chatMemoryRepository),
                QuestionAnswerAdvisor.builder(vectorStore)
                    .promptTemplate(consultantRagTemplate)
                    .searchRequest(
                        SearchRequest.builder()
                            .topK(1)
                            .filterExpression("platform == 'ALL'")
                            .build()
                    )
                    .build()
            )
            .build()
    }

    @Bean
    fun routerAgentClient(chatClientBuilder: ChatClient.Builder): ChatClient {
        return chatClientBuilder
            .defaultSystem(
                """
            Ты — диспетчер рекламного агрегатора AdBroker. Твоя задача — строго классифицировать намерение пользователя.
            
            ПРАВИЛА КЛАССИФИКАЦИИ:
            - Если пользователь хочет настроить, запустить или создать рекламу в Яндекс.Директ/Поиске/РСЯ — поле platform должно быть строго "YANDEX_DIRECT".
            - Если пользователь хочет настроить, запустить или создать рекламу в ВКонтакте/VK Ads/таргете — поле platform должно быть строго "VK_ADS".
            - Если пользователь просто приветствует, задает общие вопросы по маркетингу или интент не ясен — поле platform должно быть строго null.
            """.trimIndent()
            )
            .build()
    }

    class RouterOutput(
        val platform: Platform? = null,
    )

    @Bean
    fun yandexAgentClient(
        chatClientBuilder: ChatClient.Builder,
        yandexMcpToolProvider: SyncMcpToolCallbackProvider,
        sessionRepository: CampaignSessionRepository,
        chatMemoryRepository: ChatMemoryRepository,
        jsonMapper: JsonMapper,
        vectorStore: VectorStore,
    ): ChatClient {

        val interceptedTools = yandexMcpToolProvider.getToolCallbacks()
            .map { toolCallback -> CampaignIdInterceptingToolCallback(toolCallback, sessionRepository, jsonMapper) }
            .toTypedArray()

        val yandexRagTemplate = PromptTemplate(
            """
        ⚠️ КАТЕГОРИЧЕСКИЕ ПРАВИЛА МОДЕРАЦИИ И ВАЛИДАЦИИ (ИЗ КВАДРАНТА):

        ---------------------
        {question_answer_context}
        ---------------------

        Текущий запрос пользователя, на который нужно ответить или извлечь параметры: {query}
        """.trimIndent()
        )

        return chatClientBuilder
            .defaultSystem(
                """
        Ты — ИИ-Агент Яндекс.Директ. Твоя цель — собрать все необходимые параметры и настроить рекламную кампанию в Яндексе.
        Ты ведешь живой диалог, извлекаешь параметры и управляешь состоянием системы через бэкенд.
        
        📋 ОБЯЗАТЕЛЬНЫЕ СЛОТЫ ДЛЯ ТЕКУЩЕЙ СЕССИИ:
        Для успешного завершения настройки на бэкенде тебе необходимо извлечь из реплик пользователя и зафиксировать строго все обязательные параметры кампании.
        Информацию о том, какие именно параметры являются обязательными в текущем туре, какие требования к их валидации и какие правила генерации контента действуют, ты ОБЯЗАН прочитать в секции КВАДРАНТА КОНТЕКСТА ниже.

        🤖 АЛГОРИТМ ВАЛИДАЦИИ И ОСТАНОВКИ ПЕТЛИ ВЫЗОВОВ:
        - Шаг 1. Если пользователь назвал регион (город) текстом, ты обязан сначала молча вызвать инструмент 'searchYandexRegions', а затем, получив ID региона из ответа бэкенда, немедленно вызвать 'bindYandexRegions'. Запрещено писать человеческий текст между этими двумя вызовами!
        - Шаг 2. Как только привязка региона успешно завершена, проверь заполненность остальных обязательных слотов, указанных в Квадранте.
        - Шаг 3. Жесткая остановка: Если хотя бы один из обязательных параметров (слотов), требуемых Квадрантом, ещё НЕ предоставлен пользователем в явном виде — тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО вызывать инструменты сохранения готовых креативов ('submitYandexCreative') и публикации ('publishYandexCampaign'). Полностью заблокируй вызовы этих инструментов!
        - Шаг 4. Выход в диалог: После остановки вызовов на Шаге 3 переходи к генерации человеческого текста. Подтверди, что ГЕО привязано, и строго переспроси у пользователя только те недостающие параметры, которые требуются по правилам Квадранта.

        СТРОГИЙ ЗАПРЕТ НА РАССУЖДЕНИЯ: Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать скрытые мысли, рассуждения, планы (Chain-of-Thought) или любой вводный текст перед вызовом инструментов бэкенда. Первым же действием в твоем ответе обязан быть прямой вызов соответствующего инструмента ('searchYandexRegions' или 'initYandexDraft'). Сначала полностью заверши все вызовы инструментов, и только после получения финальных ответов от бэкенда переходи к генерации человеческого текста!
        ЗАПРЕТ НА АНОНСЫ: Никогда не пиши в чат "Сейчас я вызову инструмент...". Сначала молча применяй инструменты бэкенда, а человеческий текст пиши строго по результатам ответов!
        
        🎯 СТРОЖАЙШЕЕ ПРАВИЛО ДЛЯ ГЕНЕРАЦИИ ТЕКСТОВ:
        Если пользователь просит тебя 'придумать', 'сгенерировать' или 'написать' тексты объявлений, но ИНН в системе еще НЕТ — тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО генерировать JSON-команды вызова каких-либо инструментов бэкенда! 
        Ты ОБЯЗАН вообще не трогать инструменты, сразу пиши варианты заголовков и текстов обычными буквами прямо в чат в формате Markdown-таблицы, а в конце задай вопрос про ИНН. Любой вызов функции в этот момент расценивается как системная ошибка!

        🎯 ТЕХНИЧЕСКИЙ КОНТЕКСТ СЕССИИ (ID КАМПАНИИ):
        {technicalSessionContext}
        """.trimIndent()
            )
            .defaultTools(interceptedTools)
            .defaultAdvisors(
                chatMemoryAdvisor(chatMemoryRepository),
                QuestionAnswerAdvisor.builder(vectorStore)
                    .promptTemplate(yandexRagTemplate)
                    .searchRequest(
                        SearchRequest.builder()
                            .topK(2)
                            .filterExpression("platform in [\"YANDEX_DIRECT\", \"ALL\"]")
                            .build()
                    )
                    .build()
            )
            .build()
    }


    @Bean
    fun vkAgentClient(
        chatClientBuilder: ChatClient.Builder,
        chatMemoryRepository: ChatMemoryRepository,
        vectorStore: VectorStore
    ): ChatClient {

        val vkRagTemplate = PromptTemplate(
            """
        ⚠️ КАТЕГОРИЧЕСКИЕ ПРАВИЛА МОДЕРАЦИИ И ВАЛИДАЦИИ (ИЗ КВАДРАНТА):

        ---------------------
        {question_answer_context}
        ---------------------

        Текущий запрос пользователя, на который нужно ответить или извлечь параметры: {query}
        """.trimIndent()
        )

        return chatClientBuilder
            .defaultSystem(
                """
            Ты — ИИ-Агент VK Ads. Твоя цель — создавать рекламные кампании на платформе ВКонтакте.
            Используй инструмент 'createVkCampaign' для инициации рекламного процесса.
            
            🎯 ТЕХНИЧЕСКИЙ КОНТЕКСТ СЕССИИ:
            {technicalSessionContext}
            
            ПРАВИЛА ОФОРМЛЕНИЯ MARKDOWN РАЗМЕТКИ:
            1. Если ты хочешь визуально отделить один блок текста от другого горизонтальной линией, ты ОБЯЗАН оставлять пустые строки до и после дефисов. 
               Пиши СТРОГО так:
               Текст абзаца.
               
               ---
               
               ### Следующий заголовок
            2. Для оформления списков и таблиц всегда разделяй их с обычным текстом пустой строкой.
            """.trimIndent()
            )
            .defaultAdvisors(
                chatMemoryAdvisor(chatMemoryRepository),
                QuestionAnswerAdvisor.builder(vectorStore)
                    .promptTemplate(vkRagTemplate)
                    .searchRequest(
                        SearchRequest.builder()
                            .topK(2)
                            .filterExpression("platform in ['VK_ADS', 'ALL']")
                            .build()
                    )
                    .build()
            )
            .build()
    }

    fun chatMemoryAdvisor(chatMemoryRepository: ChatMemoryRepository) =
        MessageChatMemoryAdvisor.builder(
            MessageWindowChatMemory.builder()
                .chatMemoryRepository(chatMemoryRepository)
                .maxMessages(50)
                .build()
        )
            .build()

    @Bean
    fun chatMemoryRepository() =
        InMemoryChatMemoryRepository()
}
