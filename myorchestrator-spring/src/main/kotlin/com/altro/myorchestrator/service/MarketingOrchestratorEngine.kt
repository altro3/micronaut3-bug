package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.CampaignContext
import com.altro.myorchestrator.api.dto.Creative
import com.altro.myorchestrator.api.dto.OrchestrationState
import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors

@Service
class MarketingOrchestratorEngine(
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStore: VectorStore // Внедряем Qdrant
    // Временно убираем из конструктора McpSyncClient, так как сервера еще нет
) {
    private val log = KotlinLogging.logger {}

    fun executeOrchestration(
        briefText: String,
        statusConsumer: (OrchestrationState) -> Unit
    ): CampaignContext {
        val campaignId = UUID.randomUUID()

        statusConsumer(OrchestrationState.Initial(briefText))
        statusConsumer(OrchestrationState.Analyzing(Instant.now()))

        val chatClient = chatClientBuilder.build()

        // ШАГ 1: Анализ брифа Квеном
        val analysisResult = chatClient.prompt()
            .user("Проанализируй следующий бриф, выдели список целевых площадок (YANDEX_DIRECT, VK_ADS) и текстовое описание аудитории. Бриф: $briefText")
            .call()
            .entity(object : ParameterizedTypeReference<AnalysisResult>() {})
            ?: throw IllegalStateException("Failed to parse analysis result")

        // ШАГ 2: Векторный поиск правил модерации в Qdrant
        val moderationRulesContext = searchAdvertisingRules(
            query = "${analysisResult.audienceDescription} $briefText",
            platforms = analysisResult.recommendations.map { it.platform }
        )

        statusConsumer(
            OrchestrationState.CreativeGeneration(
                platforms = analysisResult.recommendations.map { it.platform },
                audienceDescription = analysisResult.audienceDescription
            )
        )

        // Имитируем список инструментов, который якобы пришел бы от MCP
        val mockTools = "[validateAndAddCreative(platform, creative, audience) - инструмент проверки модерации и отправки в кабинет]"

        // ШАГ 3: Генерация креативов с учетом правил из Qdrant
        val creativesTypeRef = object : ParameterizedTypeReference<Map<Platform, Creative>>() {}

        val creativePrompt = """
            На основе описания целевой аудитории: '${analysisDescription(analysisResult)}', 
            сгенерируй рекламные креативы и рассчитай бюджеты в копейках отдельно для каждой платформы: ${analysisResult.recommendations.map { it.platform }.joinToString()}. 
            
            ПРАВИЛА И ОГРАНИЧЕНИЯ ИЗ БАЗЫ ЗНАНИЙ QDRANT (Учти их, чтобы креатив прошел валидацию!):
            $moderationRulesContext
            
            Доступные инструменты автоматизации: $mockTools
        """.trimIndent()

        val creatives = chatClient.prompt()
            .user(creativePrompt)
            .call()
            .entity(creativesTypeRef)
            ?: throw IllegalStateException("Failed to generate creatives")

        statusConsumer(
            OrchestrationState.BudgetAndValidation(
                creatives = creatives,
                isApproved = true
            )
        )

        // ШАГ 4: Имитация параллельного вызова инструментов (заменяем вызов mcpSyncClient на локальный лог)
        val loomDispatcher = Executors.newVirtualThreadPerTaskExecutor().asCoroutineDispatcher()
        val campaignIds = mutableMapOf<Platform, String?>()
        val logs = CopyOnWriteArrayList<String>()

        try {
            runBlocking(loomDispatcher) {
                val jobs = analysisResult.recommendations.map { it.platform }.map { platform ->
                    async {
                        try {
                            // Вместо mcpSyncClient.callTool имитируем успешный ответ и генерируем фейковый ID кампании
                            val mockGeneratedId = "act_" + UUID.randomUUID().toString().take(8)

                            logs.add("ИМИТАЦИЯ MCP: Платформа $platform успешно прошла валидацию по правилам Qdrant.")
                            logs.add("ИМИТАЦИЯ MCP: Создана кампания в кабинете. Получен ID: $mockGeneratedId")

                            platform to mockGeneratedId
                        } catch (ex: Exception) {
                            log.error(ex) { "Ошибка при обработке платформы $platform" }
                            logs.add("Ошибка платформы $platform: ${ex.message}")
                            platform to null
                        }
                    }
                }

                jobs.awaitAll().forEach { (platform, id) ->
                    campaignIds[platform] = id
                }
            }
        } finally {
            loomDispatcher.close()
        }

        val uploadingState = OrchestrationState.Uploading(campaignIds)
        statusConsumer(uploadingState)

        val completedState = OrchestrationState.Completed(logs.toList())
        statusConsumer(completedState)

        return CampaignContext(
            campaignId = campaignId,
            state = completedState,
            updatedAt = Instant.now()
        )
    }

    private fun analysisDescription(result: AnalysisResult): String = result.audienceDescription

    private fun searchAdvertisingRules(query: String, platforms: List<Platform>): String {
        return try {
            val searchRequest = SearchRequest.builder()
                .query(query)
                .topK(3)
                .similarityThreshold(0.5) // Чуть снизим порог для тестов прототипа
                .build()

            val matchedDocs = vectorStore.similaritySearch(searchRequest)

            if (matchedDocs.isEmpty()) {
                "Локальная база правил модерации пуста. Сгенерируй стандартный качественный креатив."
            } else {
                matchedDocs.joinToString(separator = "\n\n") { doc ->
                    "[База знаний - ${doc.metadata["platform"] ?: "Общее"}]: ${doc.text}"
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Ошибка поиска в Qdrant" }
            "База знаний правил временно недоступна."
        }
    }
}
