package com.altro.myorchestrator.model

import com.altro.myorchestrator.api.dto.Platform
import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("campaign_session")
class CampaignSession(
    @Id
    var campaignId: Long = 0,                         // Локальный ID сессии чата (Primary Key)
    var currentStep: CampaignCreationStep = CampaignCreationStep.DRAFT, // Текущий шаг (старый или новый макро-статус)
    var context: SessionContext = SessionContext(),    // Контекст со всеми сквозными ID
    var updatedAt: Instant,
) {

    class SessionContext(
        var selectedPlatform: Platform? = null,        // Выбранная платформа рекламы ("YANDEX_DIRECT" или "VK_ADS")
        var yadCampaignId: Long? = null,               // Сквозной ID черновика кампании, созданный в service-yad через MCP
        var paymentInvoiceId: String? = null,          // ID счета на оплату (для финальных шагов)
        var executionLogs: List<String> = emptyList(),  // История реплик чата [Юзер, ИИ, Юзер, ИИ...], хранящая память диалога
    )
}
