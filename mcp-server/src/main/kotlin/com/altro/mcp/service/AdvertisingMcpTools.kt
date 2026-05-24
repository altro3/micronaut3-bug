package com.altro.mcp.service

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.tool.annotation.Tool
import org.springframework.stereotype.Component

@Component
class AdvertisingMcpTools {

    private val log = KotlinLogging.logger {}

    @Tool(description = "Создать рекламную кампанию на платформе ВКонтакте (VK). Требуется название, заголовок и текст баннера.")
    fun createVkCampaign(
        name: String,
        title: String,
        text: String
    ): String {
        log.info { "MCP Tool 'createVkCampaign' вызван. Параметры: name='$name', title='$title'" }

        // На этапе холостого запуска возвращаем строку.
        // В следующей итерации здесь будет вызов RestClient к вашему vk-adapter.
        return "Успешно. Кампания в VK инициирована (Холостой режим)."
    }

    @Tool(description = "Создать текстово-графическую кампанию в Яндекс.Директ. Требуется название, текст объявления и список ключевых слов.")
    fun createYandexCampaign(
        name: String,
        text: String,
        keywords: List<String>
    ): String {
        log.info { "MCP Tool 'createYandexCampaign' вызван. Параметры: name='$name', keywords=$keywords" }

        // На этапе холостого запуска возвращаем строку.
        // В следующей итерации здесь будет вызов RestClient к вашему yandex-adapter.
        return "Успешно. Кампания в Яндекс.Директ инициирована (Холостой режим)."
    }
}
