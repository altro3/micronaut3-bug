package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.CampaignContext
import com.altro.myorchestrator.api.dto.Creative
import com.altro.myorchestrator.api.dto.OrchestrationState
import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.ai.mcp.SyncMcpToolCallback
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef
import java.time.Instant
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList

@Service
class MarketingOrchestratorEngine(
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStore: VectorStore,
    private val mcpClient: McpSyncClient,
    private val jsonMapper: JsonMapper,
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
        val moderationRules = searchAdvertisingRules("правила модерации и лимиты символов " + platforms.joinToString { it.name }, platforms)

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

    private fun searchAdvertisingRules(query: String, platforms: List<Platform>): String =
        try {
            val requestBuilder = SearchRequest.builder()
                .query(query)
                .topK(3)
//                .similarityThreshold(0.5)

            if (platforms.isNotEmpty()) {
                val platformNames = platforms.map { it.name }
                val b = FilterExpressionBuilder()
                    .`in`("platform", platformNames)
                    .build()

                requestBuilder.filterExpression(b)
            }

            val searchRequest = requestBuilder.build()
            val matchedDocs = vectorStore.similaritySearch(searchRequest)

            if (matchedDocs.isEmpty()) {
                "Локальная база правил модерации пуста для платформ $platforms. Сгенерируй стандартный качественный креатив."
            } else {
                matchedDocs.joinToString(separator = "\n\n") { doc ->
                    "[База знаний - ${doc.metadata["platform"] ?: "Общее"}]: ${doc.text}"
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Ошибка поиска в Qdrant" }
            "База знаний правил временно недоступна."
        }

    private fun generateCreatives(analysisResult: AnalysisResult, moderationRules: String): Map<Platform, Creative> {
        val platformsList = analysisResult.recommendations.map { it.platform }.joinToString()

        val systemInstruction = """
            Вы — изолированный API-компонент генерации текстов. Вам ЗАПРЕЩЕНО общаться с пользователем, писать пояснения или вступления.
            Вы должны вернуть ответ СТРОГО в формате валидного JSON-объекта, соответствующего структуре.
            НЕ используйте markdown-разметку и кавычки ```json. Начните ответ сразу со знака {.
            
            Структура JSON:
            {
              "VK_ADS": {
                "title": "текст",
                "bodyText": "текст"
              },
              "YANDEX_DIRECT": {
                "title": "текст",
                "bodyText": "текст"
              }
            }
        """.trimIndent()

        val userPrompt = """
            На основе описания целевой аудитории: '${analysisResult.audienceDescription}', 
            напиши короткие рекламные тексты (заголовок title и текст объявления bodyText) отдельно для платформ: $platformsList.
            
            ПРАВИЛА МОДЕРАЦИИ ИЗ БАЗЫ ЗНАНИЙ (Обязательно примени их к текстам):
            $moderationRules
        """.trimIndent()

        log.info { "=== [DEBUG] Шаг 3: Отправка промпта в Qwen-35B. Лимит 400 токенов ===" }

        // Блокирующий вызов с жестким ограничением максимального количества токенов ответа
        val finalRawJson = chatClient.prompt()
            .system(systemInstruction)
            .user(userPrompt)
//            .options(ChatOptions.builder().maxTokens(4000))
            .call()
            .content()?.trim()
            ?: throw IllegalStateException("Модель вернула пустой ответ на Шаге 3")

        log.info { "=== [DEBUG] Шаг 3: Ответ от ИИ успешно получен ===" }
        log.info { "Сырой JSON от модели:\n$finalRawJson" }

        return try {
            jsonMapper.readValue(
                finalRawJson,
                jacksonTypeRef<Map<Platform, Creative>>(),
            )
        } catch (e: Exception) {
            log.error(e) { "Ошибка парсинга JSON на Шаге 3. Текст ответа был:\n$finalRawJson" }
            throw IllegalStateException("Модель выдала невалидную структуру JSON на Шаге 3", e)
        }
    }

    /**
     * Шаг 4: Декомпозированная публикация с real-time стримингом статусов в UI
     */
    private fun publishCampaigns(
        campaignId: UUID,
        platforms: List<Platform>,
        creatives: Map<Platform, Creative>,
        logs: CopyOnWriteArrayList<String>,
        statusConsumer: (OrchestrationState) -> Unit
    ): Map<Platform, String?> {
        val campaignIds = mutableMapOf<Platform, String?>()

        try {
            // 1. Извлекаем список инструментов напрямую через mcpClient
            val toolsResult = mcpClient.listTools()
            val availableTools = toolsResult.tools
            log.info { "Доступные в системе инструменты MCP: ${availableTools.map { it.name }}" }

            platforms.forEach { platform ->
                val creative = creatives[platform]
                if (creative != null) {
                    val toolName = when (platform) {
                        Platform.VK_ADS -> "createVkCampaign"
                        Platform.YANDEX_DIRECT -> "createYandexCampaign"
                    }

                    val startMsg = "🚀 АГЕНТ: Отправка прямого RPC-запроса в инструмент '$toolName' для $platform..."
                    logs.add(startMsg)
                    statusConsumer(OrchestrationState.Completed(logs.toList()))

                    // 2. Проверяем, что сервер 8090 отдает этот инструмент
                    if (availableTools.any { it.name == toolName }) {
                        // Строим аргументы для вызова Kotlin-метода
                        val arguments = buildMcpArguments(platform, campaignId, creative)

                        // 3. Собираем объект запроса точно по контракту твоего JSONRPCRequest
                        val request = CallToolRequest(
                            toolName,
                            arguments,
                            mapOf() // Пустая мапа метаданных _meta
                        )

                        log.info { "Физический вызов метода MCP: $toolName" }

                        // ХУЯКС! Прямой вызов метода через клиент.
                        // Никакого ИИ, чистый, быстрый и бесперебойный HTTP запрос!
                        val mcpResponse = mcpClient.callTool(request)

                        val successMsg = "✅ МСР УСПЕХ ($platform): ${mcpResponse.content}"
                        logs.add(successMsg)
                        statusConsumer(OrchestrationState.Completed(logs.toList()))

                        campaignIds[platform] = "act_" + UUID.randomUUID().toString().take(8)
                    } else {
                        val errorMsg = "❌ МСР ОШИБКА ($platform): Инструмент '$toolName' не найден на сервере 8090"
                        logs.add(errorMsg)
                        statusConsumer(OrchestrationState.Completed(logs.toList()))
                    }
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Критический сбой выполнения MCP" }
            logs.add("❌ КРИТИЧЕСКАЯ ОШИБКА АВТОМАТИЗАЦИИ: ${e.message}")
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
