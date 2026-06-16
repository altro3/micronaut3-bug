package com.altro.yad.model

import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("campaign")
data class Campaign(
    @Id
    val id: Long = 0,
    val name: String,
    val text: String,
    val targetRegionId: String,
    val externalId: Long? = null,
    val status: CampaignStatus,
    val errorMessage: String? = null,
    val createdAt: Instant = Instant.now(),
    val updatedAt: Instant = Instant.now()
)