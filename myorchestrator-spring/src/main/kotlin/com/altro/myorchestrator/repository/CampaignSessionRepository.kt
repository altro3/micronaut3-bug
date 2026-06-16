package com.altro.myorchestrator.repository

import com.altro.myorchestrator.model.CampaignSession
import org.springframework.data.jdbc.repository.query.Query
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository

@Repository
interface CampaignSessionRepository : CrudRepository<CampaignSession, Long> {

    @Query("SELECT * FROM campaign_session WHERE context->>'rawBriefText' = :firstMsg LIMIT 1")
    fun findByFirstMessage(firstMsg: String): CampaignSession?
}
