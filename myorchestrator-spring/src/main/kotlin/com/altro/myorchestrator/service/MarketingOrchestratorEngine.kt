package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.CampaignContext
import com.altro.myorchestrator.api.dto.Creative
import com.altro.myorchestrator.api.dto.OrchestrationState
import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.mcp.SyncMcpToolCallback
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList

@Service
class MarketingOrchestratorEngine(
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStore: VectorStore,
    private val mcpClient: McpSyncClient,
) {
    private val log = KotlinLogging.logger {}
    private val chatClient by lazy { chatClientBuilder.build() }

    fun executeTrueAgentOrchestration(
        userMessage: String,
        statusConsumer: (OrchestrationState) -> Unit
    ): String {
        val availableTools = mcpClient.listTools().tools

        val toolCallbacks = availableTools.map {
            SyncMcpToolCallback.builder()
                .mcpClient(mcpClient)
                .tool(it)
                .build()
        }

        statusConsumer(OrchestrationState.Analyzing(Instant.now()))

        return chatClientBuilder.build().prompt()
            .user(userMessage)
            .tools(toolCallbacks)
            .call()
            .content() ?: "Не удалось получить ответ от агента"
    }

    fun executeOrchestration(
        briefText: String,
        statusConsumer: (OrchestrationState) -> Unit
    ): CampaignContext {
        val campaignId = UUID.randomUUID()

        // Шаг 1: Анализ брифа
        statusConsumer(OrchestrationState.Initial(briefText))
        statusConsumer(OrchestrationState.Analyzing(Instant.now()))
        val analysisResult = analyzeBrief(briefText)
        val platforms = analysisResult.recommendations.map { it.platform }

        // Шаг 2: Поиск правил в Qdrant
        val moderationRules = searchAdvertisingRules("${analysisResult.audienceDescription} $briefText")

        // Шаг 3: Генерация креативов
        statusConsumer(OrchestrationState.CreativeGeneration(platforms, analysisResult.audienceDescription))
        val creatives = generateCreatives(analysisResult, moderationRules)

        // Валидация стейта для UI
        statusConsumer(OrchestrationState.BudgetAndValidation(creatives, isApproved = true))

        // Шаг 4: Публикация через adbroker-mcp-server на виртуальных потоках
        val logs = CopyOnWriteArrayList<String>()
        val campaignIds = publishCampaigns(campaignId, platforms, creatives, logs, statusConsumer)

        // Финальные статусы
        statusConsumer(OrchestrationState.Uploading(campaignIds))
        val completedState = OrchestrationState.Completed(logs.toList())
        statusConsumer(completedState)

        return CampaignContext(
            campaignId = campaignId,
            state = completedState,
            updatedAt = Instant.now()
        )
    }

    /**
     * Шаг 1: Извлечение сущностей (Платформы и Аудитория)
     */
    private fun analyzeBrief(briefText: String): AnalysisResult {
        return chatClient.prompt()
            .user("Проанализируй следующий бриф, выдели список целевых площадок (YANDEX_DIRECT, VK_ADS) и текстовое описание аудитории. Бриф: $briefText")
            .call()
            .entity(object : ParameterizedTypeReference<AnalysisResult>() {})
            ?: throw IllegalStateException("Failed to parse analysis result")
    }

    /**
     * Шаг 2: RAG поиск по правилам модерации в Qdrant
     */
    private fun searchAdvertisingRules(query: String): String =
        try {
            val searchRequest = SearchRequest.builder()
                .query(query)
                .topK(3)
                .similarityThreshold(0.5)
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

    /**
     * Шаг 3: Структурированная генерация креативов с учетом правил RAG
     */
    private fun generateCreatives(analysisResult: AnalysisResult, moderationRules: String): Map<Platform, Creative> {
        val creativePrompt = """
    На основе описания целевой аудитории: '${analysisResult.audienceDescription}', 
    сгенерируй рекламные креативы отдельно для каждой платформы: ${analysisResult.recommendations.map { it.platform }.joinToString()}. 
    
    ПРАВИЛА И ОГРАНИЧЕНИЯ ИЗ БАЗЫ ЗНАНИЙ QDRANT (ОБЯЗАТЕЛЬНО учти их при написании текстов):
    $moderationRules
    
    ВЫДАЙ ОТВЕТ СТРОГО В ФОРМАТЕ JSON. 
    Ответь ТОЛЬКО чистым JSON-объектом, НЕ оборачивай его в markdown-теги типа ```json ... ```. Начни ответ сразу с символа {.
    
    КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: Значением каждой платформы должен быть ОДИН объект JSON с ключами "title" и "bodyText".
    КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать квадратные скобки [ ] и массивы внутри платформ! Только фигурные скобки { }.
    
    Пример структуры ответа:
    {
      "VK_ADS": {
        "title": "Текст",
        "bodyText": "Текст"
      },
      "YANDEX_DIRECT": {
        "title": "Текст",
        "bodyText": "Текст"
      }
    }
""".trimIndent()

        return chatClient.prompt()
            .user(creativePrompt)
            .call()
            .entity(object : ParameterizedTypeReference<Map<Platform, Creative>>() {})
            ?: throw IllegalStateException("Failed to generate creatives")
    }

    /**
     * Шаг 4: Декомпозированная публикация с real-time стримингом статусов в UI
     */
    private fun publishCampaigns(
        campaignId: UUID,
        platforms: List<Platform>,
        creatives: Map<Platform, Creative>,
        logs: CopyOnWriteArrayList<String>,
        statusConsumer: (OrchestrationState) -> Unit // Прокидываем консьюмер статусов сюда!
    ): Map<Platform, String?> {
        val campaignIds = mutableMapOf<Platform, String?>()

        try {
            platforms.forEach { platform ->
                val creative = creatives[platform]
                if (creative != null) {
                    val startMsg = "🚀 АГЕНТ: Начинаю публикацию и вызов MCP-инструментов для $platform..."
                    logs.add(startMsg)

                    // СРАЗУ ОТПРАВЛЯЕМ СТАТУС ВО ФРОНТЕНД, чтобы текст мгновенно побежал на экране!
                    statusConsumer(OrchestrationState.Completed(logs.toList()))

                    val agentPrompt = """
                        [SYSTEM CONTEXT]
                        You are an automated API router. You cannot talk to the user. You cannot output Markdown or JSON text.
                        Your ONLY task is to call the appropriate tool for the platform $platform.
                        
                        [DATA]
                        Campaign ID: $campaignId
                        Creative Title: '${creative.title}'
                        Creative Body Text: '${creative.bodyText}'
                        
                        [CRITICAL INSTRUCTION]
                        You must IMMEDIATELY execute either 'createVkCampaign' or 'createYandexCampaign'.
                        DO NOT generate any text, explanation, or fake JSON outputs. 
                        EXECUTE THE TOOL DIRECTLY NOW.
                    """.trimIndent()

                    // Дёргаем твой агентский метод автовызова тулов
                    val agentResult = executeTrueAgentOrchestration(agentPrompt) {
                        // Холостой консьюмер для внутренних под-этапов
                    }

                    val successMsg = "✅ АГЕНТ ОТВЕТ ($platform): $agentResult"
                    logs.add(successMsg)

                    // МГНОВЕННО ОПТИМИЗИРУЕМ СТАТУС — пользователь сразу видит успешный ответ от MCP!
                    statusConsumer(OrchestrationState.Completed(logs.toList()))

                    campaignIds[platform] = "act_" + UUID.randomUUID().toString().take(8)
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Сбой автоматизации MCP" }
            logs.add("❌ Критический сбой автоматизации: ${e.message}")
            statusConsumer(OrchestrationState.Completed(logs.toList()))
        }

        return campaignIds
    }

    private fun buildMcpArguments(platform: Platform, campaignId: UUID, creative: Creative): Map<String, Any> {
        val baseName = "Кампания_${campaignId.toString().take(6)}"
        return when (platform) {
            Platform.VK_ADS -> mapOf(
                "name" to baseName,
                "title" to (creative.title ?: "Без заголовка"),
                "text" to (creative.bodyText ?: "")
            )

            Platform.YANDEX_DIRECT -> mapOf(
                "name" to baseName,
                "text" to (creative.bodyText ?: ""),
                "keywords" to listOf("купить рекламу", "локальный маркетинг")
            )
        }
    }
}
