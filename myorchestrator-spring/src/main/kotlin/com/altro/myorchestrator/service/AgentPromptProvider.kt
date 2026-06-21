package com.altro.myorchestrator.service

object AgentPromptProvider {

    private val AUTOMATION_TRIGGER_REGEX = Regex(
        "приступаю к сборке|начнем настройку|переходим к кабинету",
        RegexOption.IGNORE_CASE,
    )

    fun getDynamicTechnicalContext(role: AgentRole, campaignId: Long?, moderationRules: String): String {
        return when (role) {
            AgentRole.YANDEX_EXPERT -> """
                [ТЕХНИЧЕСКИЙ СРЕЗ ТЕКУЩЕЙ СЕССИИ (КВАДРАНТ)]
                - Числовой ID кампании в системе Postgres: ${campaignId ?: 0L}. (Если ID > 0, инструмент initYandexDraft УЖЕ выполнен успешно, тебе строго запрещено вызывать его заново! Используй этот ID для привязки регионов и остальных действий).

                [ПРАВИЛА МОДЕРАЦИИ И ЗАКОНОДАТЕЛЬСТВО ДЛЯ ГЕНЕРАЦИИ КОНТЕНТА (RAG)]
                Используй эти правила для проверки текстов объявлений и соблюдения ограничений площадки:
                $moderationRules
            """.trimIndent()

            AgentRole.VK_EXPERT -> """
                TECHNICAL CONTEXT FOR THIS TURN (VK):
                - Актуальные правила VK Ads из базы знаний (RAG): 
                $moderationRules
            """.trimIndent()

            AgentRole.CONSULTANT -> """
                CONTEXT FOR THIS TURN (CONSULTANT):
                - Полезная общая информация из базы знаний (RAG): 
                $moderationRules
            """.trimIndent()
        }
    }

    fun isTriggerTextToStartAutomation(text: String): Boolean {
        return AUTOMATION_TRIGGER_REGEX.containsMatchIn(text)
    }
}
