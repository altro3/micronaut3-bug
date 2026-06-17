package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder
import org.springframework.stereotype.Service

@Service
class VectorStoreService(
    private val vectorStore: VectorStore,
) {

    private val log = KotlinLogging.logger {}

    /**
     * Ищет правила модерации на основе выбранных платформ и текста брифа/сообщения пользователя.
     * @param platforms список целевых платформ
     * @param userBriefText текст сообщения пользователя для семантического поиска ниши бизнеса
     */
    fun searchAdvertisingRules(platforms: List<Platform>, userBriefText: String): String = try {
        // 1. Формируем поисковый запрос, подмешивая бриф для поиска по категориям бизнеса
        val finalQuery = "Правила модерации, лимиты и ограничения для бизнеса: $userBriefText"

        // Берем чуть больше документов (например, topK = 5), так как теперь ищем и лимиты, и ниши
        val requestBuilder = SearchRequest.builder().query(finalQuery).topK(5)

        if (platforms.isNotEmpty()) {
            val allowedPlatformNames = platforms.map { it.name } + "ALL"

            val filter = FilterExpressionBuilder().`in`("platform", allowedPlatformNames).build()
            requestBuilder.filterExpression(filter)
        }

        val matchedDocs = vectorStore.similaritySearch(requestBuilder.build())
        if (matchedDocs.isEmpty()) {
            "Локальная база правил пуста. Сгенерируй стандартный качественный креатив."
        } else {
            matchedDocs.joinToString(separator = "\n\n") { doc ->
                "[База знаний - ${doc.metadata["platform"] ?: "Общее"}]: ${doc.text}"
            }
        }
    } catch (e: Exception) {
        log.error(e) { "Ошибка gRPC-поиска в Qdrant" }
        "База знаний правил модерации временно недоступна."
    }
}
