package com.altro.myorchestrator.model

enum class CampaignCreationStep {
    DRAFT, // ШАГ 0: Сессия только создана, бот приветствует и просит регионы
    REGION_SELECTION,      // Шаг 1: Выбор региона из списка
    AUDIENCE_TARGETING,    // Шаг 2: Выбор и конфигурация аудитории
    CREATIVE_GENERATION,   // Шаг 3: Загрузка или ИИ-генерация креативов (тексты + картинка)
    PUBLISHING_AND_PAYMENT,// Шаг 4: Паблишинг через MCP и переход к оплате
    COMPLETED              // Финал: Кампания оплачена и активна в кабинетах
}