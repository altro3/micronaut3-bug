package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform

data class AnalysisResult(
    val recommendations: List<PlatformRecommendation>,
    val audienceDescription: String
) {
    data class PlatformRecommendation(
        val platform: Platform,
        val relevanceScore: Int, // от 0 до 100
        val reasoning: String    // почему ИИ так думает
    )
}