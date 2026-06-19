package com.altro.mcp.service

import com.altro.mcp.service.integration.serviceyad.ServiceYadClient
import com.altro.mcp.service.integration.serviceyad.dto.BindAgesRq
import com.altro.mcp.service.integration.serviceyad.dto.BindRegionsRq
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStatus
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStepRs
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStepRs.Companion.CAMPAIGN_NOT_FOUND_RS
import com.altro.mcp.service.integration.serviceyad.dto.CreateDraftRq
import com.altro.mcp.service.integration.serviceyad.dto.SubmitCreativeRq
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.model.ToolContext
import org.springframework.ai.tool.annotation.Tool
import org.springframework.stereotype.Component

@Component
class AdvertisingMcpTools(
    private val serviceYadClient: ServiceYadClient
) {

    private val log = KotlinLogging.logger {}

    @Tool(description = "Получить актуальное состояние, выбранные поля и статус синхронизации кампании напрямую из базы данных адаптера Яндекса.")
    fun getYandexCampaignStatus(toolContext: ToolContext): CampaignStepRs {
        val campaignId = extractCampaignId(toolContext) ?: return CAMPAIGN_NOT_FOUND_RS
        log.info { "MCP Tool 'getYandexCampaignStatus' запрашивает стейт для ID: $campaignId" }

        return serviceYadClient.getCampaignStatus(campaignId)
            ?: CampaignStepRs(
                campaignId = campaignId,
                status = CampaignStatus.SYNC_ERROR,
                errorMessage = "Кампания не найдена в базе адаптера"
            )
    }

    @Tool(description = "Инициировать новый черновик рекламной кампании в Яндекс.Директ. Вызывать строго один раз на самом первом шаге автоматизации Яндекса.")
    fun initYandexDraft(name: String): CampaignStepRs {
        log.info { "MCP Tool 'initYandexDraft' запускает создание кампании '$name'" }

        val response = serviceYadClient.createDraft(CreateDraftRq(name = name))
            ?: throw IllegalStateException("Внутреннее API адаптера вернуло пустой ответ при создании черновика")

        return CampaignStepRs(
            campaignId = response.campaignId,
            status = CampaignStatus.DRAFT
        )
    }

    @Tool(description = "Привязать выбранные географические регионы (список ID) к текущей кампании в Яндекс.Директ.")
    fun bindYandexRegions(regionIds: List<String>, toolContext: ToolContext): CampaignStepRs {
        val campaignId = extractCampaignId(toolContext) ?: return CAMPAIGN_NOT_FOUND_RS
        log.info { "MCP Tool 'bindYandexRegions' для ID: $campaignId" }

        serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))

        return CampaignStepRs(
            campaignId = campaignId,
            status = CampaignStatus.REGIONS_BOUND
        )
    }

    @Tool(description = "Привязать выбранные возрастные ограничения (список ID возрастных меток) к текущей кампании в Яндекс.Директ.")
    fun bindYandexAges(ageIds: List<String>, toolContext: ToolContext): CampaignStepRs {
        val campaignId = extractCampaignId(toolContext) ?: return CAMPAIGN_NOT_FOUND_RS
        log.info { "MCP Tool 'bindYandexAges' для ID: $campaignId" }

        serviceYadClient.bindAges(campaignId, BindAgesRq(ageIds = ageIds))

        return CampaignStepRs(
            campaignId = campaignId,
            status = CampaignStatus.AGES_BOUND
        )
    }

    @Tool(description = "Сохранить рекламный текст объявления в локальный черновик текущей кампании Яндекс.Директ.")
    fun submitYandexCreative(text: String, toolContext: ToolContext): CampaignStepRs {
        val campaignId = extractCampaignId(toolContext) ?: return CAMPAIGN_NOT_FOUND_RS
        log.info { "MCP Tool 'submitYandexCreative' для ID: $campaignId" }

        serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))

        return CampaignStepRs(
            campaignId = campaignId,
            status = CampaignStatus.CREATIVE_ADDED
        )
    }

    @Tool(description = "Опубликовать готовую рекламную кампанию напрямую в сеть Яндекс.Директ. Вызывать СТРОГО на самом финальном этапе.")
    fun publishYandexCampaign(toolContext: ToolContext): CampaignStepRs {
        val campaignId = extractCampaignId(toolContext) ?: return CAMPAIGN_NOT_FOUND_RS
        log.info { "MCP Tool 'publishYandexCampaign' публикует ID: $campaignId" }

        val finalResult = serviceYadClient.publishCampaign(campaignId)

        return CampaignStepRs(
            campaignId = campaignId,
            status = finalResult?.status ?: CampaignStatus.SYNCING,
            externalId = finalResult?.externalId,
            errorMessage = finalResult?.errorMessage
        )
    }

    @Tool(description = "Поиск ID географических регионов по текстовому названию.")
    fun searchYandexRegions(query: String): List<Any> {
        log.info { "MCP Tool 'searchYandexRegions' ищет: $query" }
        require(query.trim().length >= 3) { "Запрос слишком короткий. Передай минимум 3 символа." }
        return serviceYadClient.getRegions(query) ?: emptyList()
    }

    @Tool(description = "Создать рекламную кампанию на платформе ВКонтакте (VK).")
    fun createVkCampaign(name: String, title: String, text: String): CampaignStepRs {
        return CampaignStepRs(
            campaignId = 0L,
            status = CampaignStatus.DRAFT,
            errorMessage = "Холостой режим VK Ads"
        )
    }

    private fun extractCampaignId(toolContext: ToolContext): Long? {
        return when (val id = toolContext.context["campaignId"]) {
            is Long -> id
            is Int -> id.toLong()
            is String -> id.toLongOrNull()
            else -> null
        }
    }

    private fun campaignNotFoundResponse(): CampaignStepRs {
        return CampaignStepRs(
            campaignId = 0L,
            status = CampaignStatus.SYNC_ERROR,
            errorMessage = "В контексте вызова отсутствует активный campaignId. Начните с создания черновика через initYandexDraft."
        )
    }
}
