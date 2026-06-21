package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional

@Service
class SessionTransactionService(
    private val sessionRepository: CampaignSessionRepository
) {

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun saveSessionForce(session: CampaignSession): CampaignSession {
        return sessionRepository.save(session)
    }
}
