package com.altro.myorchestrator.service

import com.altro.myorchestrator.model.Platform
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

    fun searchAdvertisingRules(platforms: List<Platform>, userBriefText: String): String = try {
        val requestBuilder = SearchRequest.builder()
            .query(userBriefText)
            .topK(2)

        if (platforms.isNotEmpty()) {
            val allowedPlatformNames = platforms.map { it.name } + "ALL"
            val filter = FilterExpressionBuilder().`in`("platform", allowedPlatformNames).build()
            requestBuilder.filterExpression(filter)
        } else {
            val filter = FilterExpressionBuilder().eq("platform", "ALL").build()
            requestBuilder.filterExpression(filter)
        }

        val matchedDocs = vectorStore.similaritySearch(requestBuilder.build())
        if (matchedDocs.isEmpty()) {
            "Соблюдай стандартные лимиты длины символов и добавь маркировку рекламы с ИНН."
        } else {
            matchedDocs.joinToString(separator = "\n") { doc -> "- ${doc.text}" }
        }
    } catch (e: Exception) {
        log.error(e) { "Ошибка поиска в векторном хранилище" }
        "Обязательно добавь в текст объявления маркировку: 'Реклама. ИНН ...'"
    }
}
