package com.altro.mcp.service.integration.serviceyad

import com.altro.common.client.DefaultHttpClient
import com.altro.mcp.service.integration.serviceyad.config.ServiceYadProperties
import com.altro.mcp.service.integration.serviceyad.dto.BindAgesRq
import com.altro.mcp.service.integration.serviceyad.dto.BindRegionsRq
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStepRs
import com.altro.mcp.service.integration.serviceyad.dto.CreateDraftRq
import com.altro.mcp.service.integration.serviceyad.dto.RegionItemDto
import com.altro.mcp.service.integration.serviceyad.dto.SubmitCreativeRq
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service

@Service
class ServiceYadClient(
    props: ServiceYadProperties,
    @Qualifier("serviceYadHttpClient")
    private val httpClient: DefaultHttpClient
) {
    private val endpoints = props.endpoints

    /**
     * Шаг 1: Создать черновик кампании (Имя)
     */
    fun createDraft(request: CreateDraftRq): CampaignStepRs? =
        httpClient.sendRq(endpoints.createDraft, CampaignStepRs::class.java, request)

    /**
     * Шаг 2: Привязать выбранные ГЕО-регионы к кампании
     */
    fun bindRegions(id: Long, request: BindRegionsRq): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.bindRegions,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
            rqBody = request
        )

    /**
     * Шаг 3: Привязать выбранный возрастной таргетинг к кампании
     */
    fun bindAges(id: Long, request: BindAgesRq): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.bindAges,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
            rqBody = request
        )

    /**
     * Шаг 4: Сохранить текст объявления в локальном черновике адаптера.
     * В сеть Яндекса на этом этапе ничего не отправляется.
     */
    fun submitCreative(id: Long, request: SubmitCreativeRq): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.submitCreative,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
            rqBody = request,
        )

    /**
     * Шаг 5: Финальная публикация готовой кампании.
     * Запускает внутреннюю проверку лимитов и отправляет пакет в сеть Яндекс.Директ.
     */
    fun publishCampaign(id: Long): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.publishCampaign,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
        )

    /**
     * Дополнительный шаг: Чтение текущего состояния и технических ошибок из адаптера
     */
    fun getCampaignStatus(id: Long): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.getCampaignStatus,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
        )

    /**
     * Получить список доступных регионов из Caffeine-кэша адаптера
     */
    fun getRegions(query: String): List<RegionItemDto>? =
        httpClient.sendRq(endpoints.getRegions, RS_REGIONS, queryParams = mapOf("query" to query))

    companion object {
        val RS_REGIONS = object : ParameterizedTypeReference<List<RegionItemDto>>() {}
    }
}
