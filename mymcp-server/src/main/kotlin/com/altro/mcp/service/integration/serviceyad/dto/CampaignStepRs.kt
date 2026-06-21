package com.altro.mcp.service.integration.serviceyad.dto

data class CampaignStepRs(
    val campaignId: Long,
    val status: CampaignStatus,
    val externalId: Long? = null,
    val errorMessage: String? = null,
)
