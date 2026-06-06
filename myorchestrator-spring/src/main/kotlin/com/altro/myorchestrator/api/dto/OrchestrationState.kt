package com.altro.myorchestrator.api.dto

import java.time.Instant

/**
 * Состояние оркестрации кампании
 */
sealed interface OrchestrationState {
    /**
     * Начальное состояние с текстом брифа
     */
    data class Initial(val briefText: String) : OrchestrationState

    /**
     * Состояние анализа брифа
     */
    data class Analyzing(val startedAt: Instant) : OrchestrationState

    /**
     * Состояние генерации креативов
     */
    data class CreativeGeneration(
        val platforms: List<Platform>,
        val audienceDescription: String
    ) : OrchestrationState

    /**
     * Состояние бюджета и валидации
     */
    data class BudgetAndValidation(
        val creatives: Map<Platform, Creative>,
        val isApproved: Boolean
    ) : OrchestrationState

    /**
     * Состояние загрузки кампаний
     */
    data class Uploading(val platformCampaignIds: Map<Platform, String?>) : OrchestrationState

    /**
     * Завершенное состояние
     */
    data class Completed(val logs: List<String>) : OrchestrationState
}