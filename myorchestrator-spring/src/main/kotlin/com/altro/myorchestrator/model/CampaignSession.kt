package com.altro.myorchestrator.model

import com.altro.myorchestrator.api.dto.Platform
import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("campaign_session")
class CampaignSession(
    @Id
    var campaignId: Long = 0,
    var currentStep: CampaignCreationStep = CampaignCreationStep.DRAFT,
    var context: SessionContext = SessionContext(),
    var updatedAt: Instant = Instant.now(),
) {

    class SessionContext(
        var selectedPlatform: Platform? = null, // Храним просто строкой "YANDEX_DIRECT" или "VK_ADS"
        var yadCampaignId: Long? = null,      // Единственная сквозная ссылка на первоисточник правды в service-yad
        var paymentInvoiceId: String? = null,
        var executionLogs: List<String> = emptyList(),
    )
}
