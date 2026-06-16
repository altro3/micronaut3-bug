package com.altro.mcp.service

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.tool.annotation.Tool
import org.springframework.stereotype.Component

@Component
class AdvertisingMcpTools {

    private val log = KotlinLogging.logger {}

    /**
     * Инструмент публикации в VK
     */
    @Tool(description = "Создать рекламную кампанию на платформе ВКонтакте (VK). Требуется название, заголовок и текст баннера.")
    fun createVkCampaign(
        name: String,
        title: String,
        text: String
    ): String {
        log.info { "MCP Tool 'createVkCampaign' вызван. Параметры: name='$name', title='$title'" }
        return "Успешно. Кампания в VK инициирована (Холостой режим)."
    }

    /**
     * Инструмент публикации в Яндекс
     */
    @Tool(description = "Создать текстово-графическую кампанию в Яндекс.Директ. Требуется название, текст объявления и список ключевых слов.")
    fun createYandexCampaign(
        name: String,
        text: String,
        keywords: List<String>
    ): String {
        log.info { "MCP Tool 'createYandexCampaign' вызван. Параметры: name='$name', keywords=$keywords" }
        return "Успешно. Кампания в Яндекс.Директ инициирована (Холостой режим)."
    }

    /**
     * НОВЫЙ ИНСТРУМЕНТ: Раздача справочников регионов для оркестратора
     * Возвращает валидный JSON-массив с ID и названиями
     */
    @Tool(description = "Получить актуальный справочник доступных географических регионов для таргетинга. Требуется указать платформу (VK_ADS или YANDEX_DIRECT).")
    fun getRegionsList(platform: String): String {
        log.info { "MCP Tool 'getRegionsList' вызван для платформы: $platform" }

        // В будущем здесь будет сетевой поход в vk-adapter или yandex-adapter за реальными гео-базами.
        // Сейчас возвращаем жестко структурированный валидный JSON-массив, который Jackson-мапер оркестратора сможет распарсить.
        return when (platform.uppercase().trim()) {
            "VK_ADS" -> """
                [
                  {"id": "vk_1", "name": "Москва и Московская область"},
                  {"id": "vk_2", "name": "Санкт-Петербург и ЛО"},
                  {"id": "vk_3", "name": "Новосибирск"}
                ]
            """.trimIndent()

            "YANDEX_DIRECT" -> """
                [
                  {"id": "yam_1", "name": "Россия (Центр)"},
                  {"id": "yam_2", "name": "Россия (Сибирь)"},
                  {"id": "yam_3", "name": "Россия (Урал)"}
                ]
            """.trimIndent()

            else -> "[]"
        }
    }

    /**
     * НОВЫЙ ИНСТРУМЕНТ: Раздача возрастных маркировок и таргетингов
     * Возвращает валидный JSON-массив
     */
    @Tool(description = "Получить список доступных опций возрастных ограничений и возрастного таргетинга для выбранной платформы (VK_ADS или YANDEX_DIRECT).")
    fun getAgeTargetingOptions(platform: String): String {
        log.info { "MCP Tool 'getAgeTargetingOptions' вызван для платформы: $platform" }

        // Кабинеты требуют разные структуры маркировок (у Яндекса — жесткие дискретные метки, у ВК — ползунки от-до)
        return when (platform.uppercase().trim()) {
            "VK_ADS" -> """
                [
                  {"id": "vk_age_all", "name": "Без ограничений"},
                  {"id": "vk_age_12", "name": "Рекомендовано от 12 лет"},
                  {"id": "vk_age_18", "name": "Строго 18+"}
                ]
            """.trimIndent()

            "YANDEX_DIRECT" -> """
                [
                  {"id": "ya_age_0", "name": "0+"},
                  {"id": "ya_age_6", "name": "6+"},
                  {"id": "ya_age_12", "name": "12+"},
                  {"id": "ya_age_16", "name": "16+"},
                  {"id": "ya_age_18", "name": "18+"}
                ]
            """.trimIndent()

            else -> "[]"
        }
    }
}
