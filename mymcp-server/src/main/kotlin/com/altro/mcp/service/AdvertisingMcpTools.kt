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

    @Tool(description = "Инициировать новый черновик рекламной кампании в Яндекс.Директ. Вызывать строго на самом первом шаге, когда пользователь определился с именем кампании.")
    fun initYandexDraft(name: String): String = executeMcpStep("initYandexDraft") {
        serviceYadClient.createDraft(CreateDraftRq(name = name))
    }

    @Tool(description = "Привязать выбранные географические регионы (список ID) к существующей кампании в Яндекс.Директ. Требуется локальный числовой ID кампании.")
    fun bindYandexRegions(campaignId: Long, regionIds: List<String>): String = executeMcpStep("bindYandexRegions") {
        serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))
    }

    @Tool(description = "Привязать выбранные возрастные ограничения (список ID возрастных меток) к существующей кампании в Яндекс.Директ. Требуется локальный числовой ID кампании.")
    fun bindYandexAges(campaignId: Long, ageIds: List<String>): String = executeMcpStep("bindYandexAges") {
        serviceYadClient.bindAges(campaignId, BindAgesRq(ageIds = ageIds))
    }

    @Tool(description = "Сохранить рекламный текст объявления в локальный черновик кампании Яндекс.Директ. Этот инструмент НЕ отправляет кампанию в сеть Яндекса, а только фиксирует текст в базе. Требуется локальный числовой ID кампании.")
    fun submitYandexCreative(campaignId: Long, text: String): String = executeMcpStep("submitYandexCreative") {
        serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))
    }

    @Tool(description = "Опубликовать готовую рекламную кампанию напрямую в сеть Яндекс.Директ. Вызывать СТРОГО на самом финальном этапе, когда текст сохранен и пользователь дал явное текстовое согласие на запуск рекламы (например, сказал 'Запускай' или 'Публикуй'). Требуется локальный числовой ID кампании.")
    fun publishYandexCampaign(campaignId: Long): String = executeMcpStep("publishYandexCampaign") {
        serviceYadClient.publishCampaign(campaignId)
    }

    @Tool(description = "Поиск ID географических регионов по текстовому названию (например, 'мос', 'новосиб', 'питер'). Запрос должен быть не менее 3 символов. Обязательно вызывай этот инструмент перед привязкой регионов, чтобы узнать их точные ID.")
    fun searchYandexRegions(query: String): String {
        log.info { "MCP Tool 'searchYandexRegions' выполняет поиск по запросу: $query" }

        if (query.trim().length < 3) {
            return "{\"error\": \"Запрос слишком короткий. Передай минимум 3 символа названия региона (например, 'моск').\"}"
        }

        return try {
            val matchedRegions = serviceYadClient.getRegions(query) ?: emptyList()
            jsonMapper.writeValueAsString(matchedRegions)
        } catch (e: Exception) {
            "[]"
        }
    }

    @Tool(description = "Создать рекламную кампанию на платформе ВКонтакте (VK). Требуется название, заголовок и текст баннера.")
    fun createVkCampaign(name: String, title: String, text: String): String {
        return "Успешно. Кампания в VK Ads инициирована (Холостой режим)."
    }

    private fun <T : Any> executeMcpStep(toolName: String, action: () -> T?): String {
        return try {
            val responseBody = action()
                ?: return "❌ Ошибка serviceYad: Внутреннее API адаптера вернуло пустой ответ при вызове инструмента '$toolName'"

            jsonMapper.writeValueAsString(responseBody)
        } catch (e: Exception) {
            log.error(e) { "Критический сетевой сбой в МСР при вызове инструмента '$toolName'" }
            "❌ Сетевой критический сбой транспорта MCP -> serviceYad в инструменте '$toolName': ${e.message}"
        }
    }
}
