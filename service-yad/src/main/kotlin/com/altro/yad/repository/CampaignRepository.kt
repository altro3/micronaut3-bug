package com.altro.yad.repository

import com.altro.yad.model.Campaign
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository

@Repository
interface CampaignRepository : CrudRepository<Campaign, Long>
