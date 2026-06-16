package com.altro.yad.service.integration.yad

import com.altro.common.client.DefaultHttpClient
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRq
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRs
import com.altro.yad.service.integration.yad.dto.YadRegionsRq
import com.altro.yad.service.integration.yad.dto.YadRegionsRs
import com.altro.yad.service.integration.yad.config.YadProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class YadClient(
    props: YadProperties,
    @Qualifier("yadHttpClient")
    private val httpClient: DefaultHttpClient,
) {

    private val endpoints = props.endpoints

    fun getRegions(rq: YadRegionsRq): YadRegionsRs? =
        httpClient.sendRq(
            endpoint = endpoints.getRegions,
            responseClass = YadRegionsRs::class.java,
            rqBody = rq,
        )

    fun createCampaign(rq: YadCreateCampaignRq): YadCreateCampaignRs? =
        httpClient.sendRq(
            endpoint = endpoints.createCampaign,
            responseClass = YadCreateCampaignRs::class.java,
            rqBody = rq,
        )
}
