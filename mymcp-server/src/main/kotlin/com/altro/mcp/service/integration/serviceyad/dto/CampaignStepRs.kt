package com.altro.mcp.service.integration.serviceyad.dto

data class CampaignStepRs(
    val id: Long,
    val status: CampaignStatus,
    val externalId: Long?,
    val errorMessage: String?
)