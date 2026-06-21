package com.altro.mcp.service

import com.altro.mcp.service.integration.serviceyad.ServiceYadClient
import com.altro.mcp.service.integration.serviceyad.dto.BindAgesRq
import com.altro.mcp.service.integration.serviceyad.dto.BindRegionsRq
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStepRs
import com.altro.mcp.service.integration.serviceyad.dto.CreateDraftRq
import com.altro.mcp.service.integration.serviceyad.dto.RegionItemDto
import com.altro.mcp.service.integration.serviceyad.dto.SubmitCreativeRq
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.model.ToolContext
import org.springframework.ai.tool.annotation.Tool
import org.springframework.ai.tool.annotation.ToolParam
import org.springframework.stereotype.Component

@Component
class AdvertisingMcpTools(
    private val serviceYadClient: ServiceYadClient,
) {

    private val log = KotlinLogging.logger {}

    @Tool(description = "Создать новый черновик рекламной кампании в Яндекс.Директ. Это СТРОГО первый шаг автоматизации. Вызывай, как только пользователь выразил намерение настроить Яндекс.")
    fun initYandexDraft(
        @ToolParam(description = "Название кампании на русском языке. Если пользователь не назвал его, придумай сам на основе ниши его бизнеса.") name: String,
        toolContext: ToolContext,
    ): CampaignStepRs? {
        val sessionId = toolContext.context["sessionId"] as? String
        log.info { "🧩 [MCP] Сессия: $sessionId | Вызов initYandexDraft для: $name" }

        return serviceYadClient.createDraft(CreateDraftRq(name = name))
    }

    @Tool(description = "Привязать выбранные географические регионы к созданной кампании Яндекс.Директ. Вызывать только ПОСЛЕ того, как известен campaignId.")
    fun bindYandexRegions(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long,
        @ToolParam(description = "Список строковых ID регионов, полученных строго из searchYandexRegions.") regionIds: List<String>
    ): CampaignStepRs? {
        return serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))
    }

    @Tool(description = "Привязать возрастные ограничения целевой аудитории к созданной кампании Яндекс.Директ.")
    fun bindYandexAges(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long,
        @ToolParam(description = "Список строковых ID возрастных меток.") ageIds: List<String>
    ): CampaignStepRs? {
        return serviceYadClient.bindAges(campaignId, BindAgesRq(ageIds = ageIds))
    }

    @Tool(description = "Сохранить рекламный текст объявления в черновик кампании Яндекс.Директ.")
    fun submitYandexCreative(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long,
        @ToolParam(description = "Готовый продающий текст объявления.") text: String
    ): CampaignStepRs? {
        return serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))
    }

    @Tool(description = "Опубликовать готовую рекламную кампанию и отправить её на модерацию в сеть Яндекс.Директ. Вызывать СТРОГО после явного согласия пользователя.")
    fun publishYandexCampaign(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long
    ): CampaignStepRs? {
        return serviceYadClient.publishCampaign(campaignId)
    }

    @Tool(description = "Поиск внутренних ID географических регионов по их текстовому названию (например, 'Москва', 'Питер'). Обязательно вызывай ПЕРЕД bindYandexRegions.")
    fun searchYandexRegions(
        @ToolParam(description = "Текст запроса. Минимум 3 символа названия города (например, 'моск').") query: String
    ): List<RegionItemDto> {
        if (query.trim().length < 3) return emptyList()
        return serviceYadClient.getRegions(query) ?: emptyList()
    }

    @Tool(description = "Создать новую рекламную кампанию на платформе ВКонтакте (VK Ads).")
    fun createVkCampaign(
        @ToolParam(description = "Название рекламной кампании.") name: String,
        @ToolParam(description = "Короткий заголовок объявления.") title: String,
        @ToolParam(description = "Основной рекламный текст для баннера.") text: String
    ): String {
        return "Успешно. Кампания в VK Ads инициирована (Холостой режим)."
    }
}
