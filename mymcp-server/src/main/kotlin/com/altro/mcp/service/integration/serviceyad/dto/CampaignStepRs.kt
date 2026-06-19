package com.altro.mcp.service.integration.serviceyad.dto

data class CampaignStepRs(
    val campaignId: Long,
    val status: CampaignStatus,
    val externalId: Long? = null,
    val errorMessage: String? = null,
    val errorAction: String? = null
) {

    companion object {
        val CAMPAIGN_NOT_FOUND_RS = CampaignStepRs(
            campaignId = 0L,
            status = CampaignStatus.SYNC_ERROR,
            errorMessage = "В контексте вызова отсутствует активный campaignId. Начните с создания черновика через initYandexDraft.",
            errorAction = "CALL_TOOL:initYandexDraft",
        )
    }
}
