package com.altro.myorchestrator.service.steps.handlers

import com.altro.myorchestrator.domain.CampaignCreationStep
import com.altro.myorchestrator.domain.CampaignSession
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.repository.CampaignSessionRepository
import org.springframework.stereotype.Component
import reactor.core.publisher.FluxSink
import java.time.Instant

@Component
class RegionSelectionHandler(private val sessionRepository: CampaignSessionRepository) {

    fun processRegions(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        val regions = userInput.split(",").map { it.trim() }
        
        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.REGION_SELECTION,
                context = session.context.copy(selectedRegions = regions),
                updatedAt = Instant.now()
            )
        )

        sink.next("✅ Регионы для площадки ${session.context.selectedPlatform?.name} сохранены: ${regions.joinToString()}.\n\n")
        sink.next("🤖 **[Шаг 3/5]**: Отправьте мне **текстовый бриф** продукта под эту аудиторию:")
        sink.complete()
    }
}
