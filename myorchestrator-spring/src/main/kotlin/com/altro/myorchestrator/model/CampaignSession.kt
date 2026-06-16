package com.altro.myorchestrator.model

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.service.AnalysisResult
import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("campaign_session")
data class CampaignSession(
    @Id
    val campaignId: Long = 0,
    val currentStep: CampaignCreationStep,
    val context: SessionContext,
    val updatedAt: Instant,
) {

    data class SessionContext(
        val selectedPlatform: Platform? = null,

        val campaignData: MarketingCampaignData = MarketingCampaignData(),

        val mcpAvailableRegions: List<McpDictionaryItem> = emptyList(),
        val mcpAvailableAges: List<McpDictionaryItem> = emptyList(),

        val rawBriefText: String? = null,
        val analysisResult: AnalysisResult? = null,
        val moderationRules: String? = null,
        val generatedCreatives: Map<Platform, CreativeWithImage>? = null,
        val executionLogs: List<String> = emptyList(),
        val platformCampaignIds: Map<Platform, String?> = emptyMap(),
        val paymentInvoiceId: String? = null
    )

    data class MarketingCampaignData(
        val name: String? = null,
        val targetRegionIds: List<String> = emptyList(),
        val minAge: Int? = null,
        val maxAge: Int? = null,
        val budgetLimit: Double? = null
    )

    data class McpDictionaryItem(
        val id: String,
        val name: String,
    )

    data class CreativeWithImage(
        val title: String,
        val bodyText: String,
        val imageUrl: String? = null, // Ссылка на локальное хранилище / S3 со сгенерированной картинкой
    )
}
