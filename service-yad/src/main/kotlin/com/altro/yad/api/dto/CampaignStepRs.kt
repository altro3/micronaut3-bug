package com.altro.yad.api.dto

import com.altro.yad.model.CampaignStatus

data class CampaignStepRs(
    val id: Long,
    val status: CampaignStatus,
    val externalId: Long? = null,
    val errorMessage: String? = null,
)