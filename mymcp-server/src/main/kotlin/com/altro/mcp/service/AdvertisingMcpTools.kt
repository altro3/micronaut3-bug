package com.altro.mcp.service

import com.altro.mcp.service.integration.serviceyad.ServiceYadClient
import com.altro.mcp.service.integration.serviceyad.dto.BindAgesRq
import com.altro.mcp.service.integration.serviceyad.dto.BindRegionsRq
import com.altro.mcp.service.integration.serviceyad.dto.CreateDraftRq
import com.altro.mcp.service.integration.serviceyad.dto.SubmitCreativeRq
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.tool.annotation.Tool
import org.springframework.stereotype.Component
import tools.jackson.databind.json.JsonMapper

@Component
class AdvertisingMcpTools(
    private val serviceYadClient: ServiceYadClient,
    private val jsonMapper: JsonMapper
) {

    private val log = KotlinLogging.logger {}

    /**
     * МСР-Инструмент Инспекции: Получить текущее состояние кампании из адаптера Яндекса
     */
    @Tool(description = "Получить актуальное состояние, выбранные поля и статус синхронизации кампании напрямую из базы данных адаптера Яндекса. Требуется локальный числовой ID кампании.")
    fun getYandexCampaignStatus(campaignId: Long): String {
        log.info { "MCP Tool 'getYandexCampaignStatus' запрашивает стейт для ID: $campaignId" }
        return try {
            val res = serviceYadClient.getCampaignStatus(campaignId)
                ?: return "{\"status\": \"NOT_FOUND\", \"errorMessage\": \"Кампания не найдена в базе адаптера\"}"
            jsonMapper.writeValueAsString(res)
        } catch (e: Exception) {
            "{\"status\": \"TRANSPORT_ERROR\", \"errorMessage\": \"Сетевой сбой при чтении стейта: ${e.message}\"}"
        }
    }

    /**
     * МСР-Инструмент Шага 1: Инициация черновика в service-yad
     */
    @Tool(description = "Инициировать новый черновик рекламной кампании в Яндекс.Директ. Вызывать строго на самом первом шаге, когда пользователь определился с именем кампании.")
    fun initYandexDraft(name: String): String = executeMcpStep("initYandexDraft") {
        serviceYadClient.createDraft(CreateDraftRq(name = name))
    }

    /**
     * МСР-Инструмент Шага 2: Привязка регионов таргетинга в service-yad
     */
    @Tool(description = "Привязать выбранные географические регионы (список ID) к существующей кампании в Яндекс.Директ. Требуется локальный числовой ID кампании.")
    fun bindYandexRegions(campaignId: Long, regionIds: List<String>): String = executeMcpStep("bindYandexRegions") {
        serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))
    }

    /**
     * МСР-Инструмент Шага 3: Привязка возрастного таргетинга в service-yad
     */
    @Tool(description = "Привязать выбранные возрастные ограничения (список ID возрастных меток) к существующей кампании в Яндекс.Директ. Требуется локальный числовой ID кампании.")
    fun bindYandexAges(campaignId: Long, ageIds: List<String>): String = executeMcpStep("bindYandexAges") {
        serviceYadClient.bindAges(campaignId, BindAgesRq(ageIds = ageIds))
    }

    /**
     * МСР-Инструмент Шага 4: Добавление креатива, валидация и автоматический уход в сеть Яндекса
     */
    @Tool(description = "Отправить финальный рекламный текст объявления в кампанию Яндекс.Директ. Этот инструмент запускает автоматическую внутреннюю валидацию лимитов и синхронизирует кампанию с сетью.")
    fun submitYandexCreative(campaignId: Long, text: String): String = executeMcpStep("submitYandexCreative") {
        serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))
    }

    /**
     * МСР-Инструмент получения справочника ГЕО-регионов
     */
    @Tool(description = "Получить актуальный справочник доступных географических регионов для таргетинга в Яндекс.Директ.")
    fun getRegionsList(): String {
        log.info { "MCP Tool 'getRegionsList' запрашивает кэшированную коллекцию из serviceYad" }
        return try {
            val regionsList = serviceYadClient.getRegions() ?: emptyList()
            jsonMapper.writeValueAsString(regionsList)
        } catch (e: Exception) {
            "[]"
        }
    }

    /**
     * Холостой инструмент для VK Ads
     */
    @Tool(description = "Создать рекламную кампанию на платформе ВКонтакте (VK). Требуется название, заголовок и текст баннера.")
    fun createVkCampaign(name: String, title: String, text: String): String {
        return "Успешно. Кампания в VK Ads инициирована (Холостой режим)."
    }

    // ==========================================================================================
    // ВЫДЕЛЕННЫЙ ОБЩИЙ МЕТОД ОБРАБОТКИ СЕТЕВЫХ ШАГОВ
    // ==========================================================================================

    /**
     * Универсальный инкапсулятор сетевого транспорта и Jackson-сериализации ответов
     */
    private fun <T : Any> executeMcpStep(toolName: String, action: () -> T?): String {
        return try {
            val responseBody = action()
                ?: return "❌ Ошибка serviceYad: Внутреннее API адаптера вернуло пустой ответ при вызове инструмента '$toolName'"

            // Если вызов успешен — намертво пакуем DTO в JSON-строку для ИИ-модели
            jsonMapper.writeValueAsString(responseBody)
        } catch (e: Exception) {
            log.error(e) { "Критический сетевой сбой в МСР при вызове инструмента '$toolName'" }
            "❌ Сетевой критический сбой транспорта MCP -> serviceYad в инструменте '$toolName': ${e.message}"
        }
    }
}
