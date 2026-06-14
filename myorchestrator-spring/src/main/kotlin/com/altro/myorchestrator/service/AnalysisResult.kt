package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform

data class AnalysisResult(
    val recommendations: List<PlatformRecommendation>,
    val audienceDescription: String
) {
    data class PlatformRecommendation(
        val platform: Platform,
        val relevanceScore: Int,
        val reasoning: String,
    )
}