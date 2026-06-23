package com.altro.myorchestrator.config

import com.altro.myorchestrator.model.Platform
import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CampaignIdInterceptingToolCallback
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor
import org.springframework.ai.chat.memory.ChatMemoryRepository
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.chat.prompt.PromptTemplate
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider
import org.springframework.ai.openai.OpenAiChatModel.ResponseFormat
import org.springframework.ai.openai.OpenAiChatModel.ResponseFormat.Type.JSON_OBJECT
import org.springframework.ai.openai.OpenAiChatOptions
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.Ordered
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
                SimpleLoggerAdvisor.builder().build(),
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
            .defaultOptions(
                OpenAiChatOptions.builder()
                    .responseFormat(ResponseFormat.builder().type(JSON_OBJECT).build())
                    .parallelToolCalls(false)
            )
            .defaultAdvisors(
                SimpleLoggerAdvisor.builder().build(),
            )
            .defaultSystem(
                """
            Ты — диспетчер рекламного агрегатора AdBroker. Твоя задача — строго классифицировать намерение пользователя.
            
            ПРАВИЛА КЛАССИФИКАЦИИ:
            - Если пользователь хочет настроить, запустить или создать рекламу в Яндекс.Директ/Поиске/РСЯ — поле platform должно быть строго "YANDEX_DIRECT".
            - Если пользователь хочет настроить, запустить или создать рекламу в ВКонтакте/VK Ads/таргете — поле platform должно быть строго "VK_ADS".
            - Если пользователь просто приветствует, задает общие вопросы по маркетингу или интент не ясен — поле platform должно быть строго null.
            
            КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ФОРМАТУ:
            Не используй теги <function_call> или инструменты. Выдай ответ СТРОГО в формате валидного JSON-объекта, без Markdown разметки (без ```json).
            Пример ответа: {"platform": "YANDEX_DIRECT"} или {"platform": null}
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
            .defaultTools(interceptedTools)
            .defaultAdvisors(
                chatMemoryAdvisor(chatMemoryRepository),
                SimpleLoggerAdvisor.builder().build(),
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
                .maxMessages(30)
                .build()
        )
            .build()

    @Bean
    fun chatMemoryRepository() =
        InMemoryChatMemoryRepository()
}
