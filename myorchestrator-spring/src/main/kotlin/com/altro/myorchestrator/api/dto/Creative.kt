package com.altro.myorchestrator.api.dto

/**
 * Креатив для рекламной кампании
 */
data class Creative(
    val title: String,
    val bodyText: String,
    val budgetKopecks: Long
)