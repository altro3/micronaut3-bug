package com.altro.myorchestrator.api.dto

import java.time.Instant
import java.util.UUID

/**
 * Контекст кампании с состоянием оркестрации
 */
data class CampaignContext(
    val campaignId: UUID,
    val state: OrchestrationState,
    val updatedAt: Instant
)