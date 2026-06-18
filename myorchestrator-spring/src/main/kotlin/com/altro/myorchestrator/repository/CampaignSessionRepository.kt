package com.altro.myorchestrator.repository

import com.altro.myorchestrator.model.CampaignSession
import org.springframework.data.jdbc.repository.query.Query
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository
import java.util.UUID

@Repository
interface CampaignSessionRepository : CrudRepository<CampaignSession, UUID>
