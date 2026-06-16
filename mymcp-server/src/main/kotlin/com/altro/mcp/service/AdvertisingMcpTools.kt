package com.altro.mcp.service

import com.altro.mcp.service.integration.serviceyad.ServiceYadClient
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
     * МСР-Инструмент Шага 1: Инициация черновика в service-yad
     */
    @Tool(description = "Инициировать новый черновик рекламной кампании в Яндекс.Директ. Вызывать строго на самом первом шаге, когда пользователь определился с именем кампании.")
    fun initYandexDraft(name: String): String {
        log.info { "MCP Tool 'initYandexDraft' вызван для имени: $name" }
        return try {
            val res = serviceYadClient.createDraft(CreateDraftRq(name = name))
                ?: return "❌ Ошибка serviceYad: не удалось создать черновик"

            jsonMapper.writeValueAsString(res)
        } catch (e: Exception) {
            "❌ Сетевой сбой при инициации черновика в serviceYad: ${e.message}"
        }
    }

    /**
     * МСР-Инструмент Шага 2: Привязка регионов таргетинга в service-yad
     */
    @Tool(description = "Привязать выбранные географические регионы (список ID) к существующей кампании в Яндекс.Директ. Требуется локальный числовой ID кампании.")
    fun bindYandexRegions(campaignId: Long, regionIds: List<String>): String {
        log.info { "MCP Tool 'bindYandexRegions' вызван для ID $campaignId и регионов $regionIds" }
        return try {
            val res = serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))
                ?: return "❌ Ошибка serviceYad: не удалось привязать регионы"

            jsonMapper.writeValueAsString(res)
        } catch (e: Exception) {
            "❌ Сетевой сбой при привязке регионов в serviceYad: ${e.message}"
        }
    }

    /**
     * МСР-Инструмент Шага 4: Добавление креатива, валидация и автоматический уход в сеть
     */
    @Tool(description = "Отправить финальный рекламный текст объявления в кампанию Яндекс.Директ. Этот инструмент запускает автоматическую внутреннюю валидацию лимитов и синхронизирует кампанию с сетью.")
    fun submitYandexCreative(campaignId: Long, text: String): String {
        log.info { "MCP Tool 'submitYandexCreative' вызван для ID $campaignId" }
        return try {
            val res = serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))
                ?: return "❌ Ошибка serviceYad: не удалось отправить объявление"

            jsonMapper.writeValueAsString(res)
        } catch (e: Exception) {
            "❌ Сетевой сбой при отправке креатива в serviceYad: ${e.message}"
        }
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
}
