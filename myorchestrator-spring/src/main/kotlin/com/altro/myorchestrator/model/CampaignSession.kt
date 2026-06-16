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
        // Шаг 1: Гео
        val selectedRegions: List<String> = emptyList(),

        // Шаг 2: Целевая аудитория
        val rawBriefText: String? = null,
        val targetAudienceDescription: String? = null,
        val analysisResult: AnalysisResult? = null, // Твой DTO из Части 1

        // Шаг 3: Креативы (Тексты + Картинки)
        val moderationRules: String? = null,
        val generatedCreatives: Map<Platform, CreativeWithImage>? = null,

        // Шаг 4: MCP Паблишинг и логи для биллинга
        val executionLogs: List<String> = emptyList(),
        val platformCampaignIds: Map<Platform, String?> = emptyMap(),
        val paymentInvoiceId: String? = null
    )

    data class CreativeWithImage(
        val title: String,
        val bodyText: String,
        val imageUrl: String? = null, // Ссылка на локальное хранилище / S3 со сгенерированной картинкой
    )
}
