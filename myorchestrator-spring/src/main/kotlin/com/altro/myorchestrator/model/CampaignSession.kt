package com.altro.myorchestrator.model

import com.altro.myorchestrator.api.dto.Platform
import org.springframework.data.annotation.Id
import org.springframework.data.annotation.Transient
import org.springframework.data.domain.Persistable
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant
import java.util.UUID

@Table("campaign_session")
class CampaignSession(
    @Id
    private var id: UUID? = null,
    var platform: Platform? = null,
    var campaignId: Long? = null,
    var context: SessionContext = SessionContext(),
    var updatedAt: Instant = Instant.now(),
) : Persistable<UUID> {

    @Transient
    var isNewEntity: Boolean = false

    override fun getId(): UUID? = id

    override fun isNew(): Boolean = isNewEntity

    class SessionContext(
        var rawBriefText: String? = null,
        var executionLogs: List<String> = emptyList(),
    )
}
