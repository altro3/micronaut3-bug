package com.altro.myorchestrator.config

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class AiConfig {

    @Bean("consultantAgentClient")
    fun consultantAgentClient(chatClientBuilder: ChatClient.Builder): ChatClient {
        return chatClientBuilder
            .defaultSystem("""
            Ты — опытный, дружелюбный и проактивный ИИ-маркетолог рекламного агрегатора AdBroker.
            Твоя цель — вести свободный диалог с пользователем, отвечать на его вопросы про маркетинг, помогать придумывать стратегии и анализировать ниши.
            
            ПРАВИЛА ПОВЕДЕНИЯ:
            1. Если пользователь просто приветствует тебя или общается на свободные темы — поддерживай диалог как эксперт.
            2. Если пользователь сомневается, какую сеть выбрать, объясни разницу: Яндекс.Директ хорош для горячего спроса (когда ищут автосервис прямо сейчас), а VK Ads — для прогрева аудитории через визуальный контент и сообщества.
            3. Как только в процессе разговора пользователь четко скажет: "Давай настраивать Яндекс" или "Давай сделаем кампанию в ВК", вежливо зафиксируй это и передай, что ты готов начать.
        """.trimIndent())
            .build()
    }

    // 1. Биг-Босс: ИИ-Роутер (Ему инструменты вообще не нужны, он просто классифицирует текст)
    @Bean("routerAgentClient")
    fun routerAgentClient(chatClientBuilder: ChatClient.Builder): ChatClient {
        return chatClientBuilder
            .defaultSystem(
                """
                Ты — диспетчер рекламного агрегатора. Твоя задача — понять, какую платформу выбрал юзер.
                Если речь про Яндекс, Директ, Поиск — ответь словом: YANDEX.
                If про ВК, ВКонтакте, VK, таргет, паблики — ответь словом: VK.
                Если интент не ясен — ответь словом: UNKNOWN.
                Будь лаконичен, пиши только одно слово.
            """.trimIndent()
            )
            .build()
    }

    // 2. Изолированный Агент Яндекса (Видит только SyncMcpToolCallbackProvider)
    @Bean("yandexAgentClient")
    fun yandexAgentClient(
        chatClientBuilder: ChatClient.Builder,
        yandexMcpToolProvider: SyncMcpToolCallbackProvider
    ): ChatClient {
        return chatClientBuilder
            // Привязываем сетевые инструменты Яндекса, которые отдал MCP сервер
            .defaultTools(yandexMcpToolProvider)
            .build()
    }

    // 3. Изолированный Агент ВК
    @Bean("vkAgentClient")
    fun vkAgentClient(chatClientBuilder: ChatClient.Builder): ChatClient {
        return chatClientBuilder
            // Если для ВК будет отдельный MCP провайдер, привяжешь его сюда.
            // Пока оставляем чистым или вешаем локальные инструменты
            .build()
    }

}
