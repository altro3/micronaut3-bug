package com.altro.yad.model

import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("campaign")
class Campaign(
    @Id
    var id: Long = 0,
    var status: CampaignStatus = CampaignStatus.DRAFT,
    var externalId: Long? = null,
    var errorMessage: String? = null,
    var data: CampaignData = CampaignData(),
    var createdAt: Instant,
    var updatedAt: Instant,
) {

    class CampaignData(
        var name: String? = null,
        var text: String? = null,
        var targetRegionIds: List<String> = emptyList(),
        var targetAgeIds: List<String> = emptyList(),
        var budgetLimit: Double? = null,
    )
}