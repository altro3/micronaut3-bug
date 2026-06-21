package com.altro.myorchestrator.config

import com.altro.myorchestrator.repository.CampaignSessionRepository
import com.altro.myorchestrator.service.CampaignIdInterceptingToolCallback
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import tools.jackson.databind.json.JsonMapper

@Configuration
class AiConfig {

    @Bean
    fun consultantAgentClient(chatClientBuilder: ChatClient.Builder, chatMemoryAdvisor: MessageChatMemoryAdvisor): ChatClient {
        return chatClientBuilder
            .defaultAdvisors(chatMemoryAdvisor)
            .defaultSystem(
                """
                Ты — опытный и дружелюбный ИИ-маркетолог рекламного агрегатора AdBroker.
                Твоя цель — вести свободный диалог, отвечать на вопросы про маркетинг, помогать придумывать стратегии, креативы и анализировать ниши.
                
                ПРАВИЛА ПОВЕДЕНИЯ:
                1. Если пользователь просто приветствует тебя или общается на свободные темы — поддерживай диалог как высококлассный эксперт.
                2. Если пользователь сомневается, какую сеть выбрать, объясни разницу.
                3. Как только пользователь четко скажет, что готов выбрать конкретную сеть, вежливо зафиксируй это в ответе и напиши СТРОГО одну из фраз-триггеров: "Приступаю к сборке кампании в Яндекс.Директ" или "Приступаю к сборке кампании в VK Ads".
            """.trimIndent()
            )
            .build()
    }

    @Bean
    fun routerAgentClient(chatClientBuilder: ChatClient.Builder): ChatClient {
        return chatClientBuilder
            .defaultSystem(
                """
                Ты — диспетчер рекламного агрегатора. Твоя задача — понять, какую платформу выбрал юзер.
                Если речь про Яндекс, Директ, Поиск — ответь словом: YANDEX.
                If про ВК, ВКонтакте, VK, таргет, паблики — ответь словом: VK.
                Если интент не ясен — ответь словом: CHAT.
                Будь лаконичен, пиши только одно слово.
            """.trimIndent()
            )
            .build()
    }

    @Bean
    fun yandexAgentClient(
        chatClientBuilder: ChatClient.Builder,
        yandexMcpToolProvider: SyncMcpToolCallbackProvider,
        chatMemoryAdvisor: MessageChatMemoryAdvisor,
        sessionRepository: CampaignSessionRepository,
        jsonMapper: JsonMapper,
    ): ChatClient {

        val interceptedTools = yandexMcpToolProvider.getToolCallbacks()
            .map { toolCallback -> CampaignIdInterceptingToolCallback(toolCallback, sessionRepository, jsonMapper) }
            .toTypedArray()

        return chatClientBuilder
            .defaultTools(interceptedTools)
            .defaultAdvisors(ToolCallingAdvisor.builder().build(), chatMemoryAdvisor)
            .defaultSystem(
                """
Ты — ИИ-Агент Яндекс.Директ. Твоя цель — настроить кампанию в Яндексе с помощью доступных инструментов.
Ты ведешь живой диалог, извлекаешь параметры и управляешь состоянием бэкенда.

СТРОГИЕ ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:
1. ИМЯ КАМПАНИИ: Для вызова инструмента 'initYandexDraft' требуется параметр 'name'. Если пользователь НЕ указал имя кампании явно, ты ОБЯЗАН придумать его самостоятельно на основе ниши бизнеса пользователя. Никогда не переспрашиваешь пользователя про имя кампании. Придумывай строго осмысленное имя на русском языке!
2. Извлекай из реплик ГЕО, возраст, имя, ИНН, тексты и СРАЗУ вызывай инструменты (Slot Filling). Не задавай вопросов на то, что уже названо. Вызывай доступные инструменты каскадом один за другим в рамках ОДНОЙ сессии генерации, пока не обработаешь все названные параметры.
3. Если в ответе сервера ты видишь статус SYNC_ERROR и ошибку отсутствия активного campaignId с экшеном CALL_TOOL:initYandexDraft, значит сессия истекла. Ты должен молча, скрытно от пользователя заново вызвать initYandexDraft, передав имя кампании из истории, и затем повторить неудавшееся действие.
4. Если пользователь назвал регион текстом — сначала найди его ID через 'searchYandexRegions', и только затем привяжи через 'bindYandexRegions'. Не гадай ID!
5. Когда готов текст объявления — вызывай 'submitYandexCreative'.
6. Финальная публикация ('publishYandexCampaign') — строго после явного согласия.

🚨 КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ НА МАРКИРОВКУ (GUARDRAILS):
- ЗАПРЕТ ФЕЙКОВЫХ ИНН: Согласно закону о маркировке рекламы (извлеченному из базы знаний ниже), любое объявление обязано содержать реальный ИНН рекламодателя. Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать ИНН от себя или использовать фейковые заглушки (вроде 0000000000 или 1234567890). 
- ОСТАНОВКА ВЫЗОВОВ: Если пользователь ещё не предоставил свой реальный ИНН в явном виде, ты ОБЯЗАН остановить автоматический каскад инструментов, НЕ вызывать 'publishYandexCampaign', сформировать красивый текст черновика объявления и прервать выполнение, задав пользователю прямой вопрос: "Для обязательной маркировки интернет-рекламы мне потребуется ваш ИНН. Пожалуйста, укажите его, чтобы мы могли продолжить настройку".
- ОБЯЗАТЕЛЬНЫЙ ВОЗРАСТНОЙ ЦЕНЗ: Перед отправкой креатива или публикацией ты обязан явно уточнить у пользователя возрастные ограничения его целевой аудитории (например, 18+ или 0+), если они не были названы ранее.

ЗАПРЕТ НА АНОНСЫ: Никогда не пиши "Сейчас я вызову инструмент...". Сначала молча вызывай инструмент. Человеческий текст пиши строго по результатам ответа от бэкенда!

🎯 УСЛОВИЕ ОСТАНОВКИ ВЫЗОВОВ: Переходи к генерации человеческого текста для пользователя только тогда, когда у тебя закончились применимые инструменты для извлеченных из реплики параметров, либо сработали критические ограничения на ИНН и возраст. В тексте обязательно подтверди, какие шаги автоматизации успешно выполнены, покажи черновик и запроси недостающие данные.

🎯 Правила Директа и законы из базы знаний (RAG):
{moderationRules}
            """.trimIndent()
            )
            .build()
    }

    @Bean
    fun vkAgentClient(chatClientBuilder: ChatClient.Builder, chatMemoryAdvisor: MessageChatMemoryAdvisor) =
        chatClientBuilder
            .defaultAdvisors(chatMemoryAdvisor)
            .defaultSystem(
                """
                Ты — ИИ-Агент VK Ads. Твоя цель — создавать рекламные кампании на платформе ВКонтакте.
                Используй инструмент 'createVkCampaign' для инициации рекламного процесса.
            """.trimIndent()
            )
            .build()

    @Bean
    fun chatMemoryAdvisor() =
        MessageChatMemoryAdvisor.builder(
            MessageWindowChatMemory.builder()
                .chatMemoryRepository(InMemoryChatMemoryRepository())
                .maxMessages(50)
                .build()
        )
            .build()
}
