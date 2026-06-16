package com.altro.mcp.service.integration.serviceyad

import com.altro.common.client.DefaultHttpClient
import com.altro.mcp.service.integration.serviceyad.config.ServiceYadProperties
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

    fun createDraft(request: CreateDraftRq): CampaignStepRs? =
        httpClient.sendRq(endpoints.createDraft, CampaignStepRs::class.java, request)

    fun bindRegions(id: Long, request: BindRegionsRq): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.bindRegions,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
            rqBody = request
        )

    fun submitCreative(id: Long, request: SubmitCreativeRq): CampaignStepRs? =
        httpClient.sendRq(
            endpoint = endpoints.submitCreative,
            responseClass = CampaignStepRs::class.java,
            pathVars = mapOf("id" to id),
            rqBody = request,
        )

    fun getRegions(): List<RegionItemDto>? =
        httpClient.sendRq(endpoints.getRegions, RS_REGIONS)

    companion object {

        val RS_REGIONS = object : ParameterizedTypeReference<List<RegionItemDto>>() {}
    }
}
