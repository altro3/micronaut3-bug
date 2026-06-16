package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder
import org.springframework.stereotype.Service

@Service
class VectorStoreService(private val vectorStore: VectorStore) {
    private val log = KotlinLogging.logger {}

    fun searchAdvertisingRules(platforms: List<Platform>): String = try {
        val query = "правила модерации и лимиты символов " + platforms.joinToString { it.name }
        val requestBuilder = SearchRequest.builder().query(query).topK(3)
        
        if (platforms.isNotEmpty()) {
            val filter = FilterExpressionBuilder().`in`("platform", platforms.map { it.name }).build()
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
