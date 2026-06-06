package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.CampaignContext
import com.altro.myorchestrator.api.dto.Creative
import com.altro.myorchestrator.api.dto.OrchestrationState
import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.springframework.ai.chat.client.ChatClient
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors

//@Service
class MarketingOrchestratorEngine(
    private val chatClientBuilder: ChatClient.Builder,
    private val mcpSyncClient: McpSyncClient
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

        val analysisResult = chatClient.prompt()
            .user("Проанализируй следующий бриф, выдели список целевых площадок (YANDEX_DIRECT, VK_ADS) и текстовое описание аудитории. Бриф: $briefText")
            .call()
            .entity(object : ParameterizedTypeReference<AnalysisResult>() {})
            ?: throw IllegalStateException("Failed to parse analysis result")

        statusConsumer(
            OrchestrationState.CreativeGeneration(
                platforms = analysisResult.recommendations.map { it.platform },
                audienceDescription = analysisResult.audienceDescription
            )
        )

        val tools = mcpSyncClient.listTools()

        val creativesTypeRef = object : ParameterizedTypeReference<Map<Platform, Creative>>() {}
        val creatives = chatClient.prompt()
            .user("На основе описания целевой аудитории: '${analysisResult.audienceDescription}', сгенерируй рекламные креативы и рассчитай бюджеты в копейках отдельно для каждой платформы: ${analysisResult.recommendations.map { it.platform }.joinToString()}. Используй доступные инструменты: $tools")
            .call()
            .entity(creativesTypeRef)
            ?: throw IllegalStateException("Failed to generate creatives")

        statusConsumer(
            OrchestrationState.BudgetAndValidation(
                creatives = creatives,
                isApproved = true
            )
        )

        val loomDispatcher = Executors.newVirtualThreadPerTaskExecutor().asCoroutineDispatcher()
        val campaignIds = mutableMapOf<Platform, String?>()
        val logs = CopyOnWriteArrayList<String>()

        try {
            runBlocking(loomDispatcher) {
                val jobs = analysisResult.recommendations.map { it.platform }.map { platform ->
                    async {
                        try {
                            val creative = creatives[platform] ?: throw IllegalStateException("Creative not found for platform: $platform")
                            val result = mcpSyncClient.callTool(
                                McpSchema.CallToolRequest(
                                    "validateAndAddCreative",
                                    mapOf(
                                        "platform" to platform.name,
                                        "creative" to creative,
                                        "audience" to analysisResult.audienceDescription
                                    )
                                )
                            )
                            logs.add("Успешно обработана платформа: $platform")
                            platform to result.content().firstOrNull()?.toString()
                        } catch (ex: Exception) {
                            log.error(ex) { "Ошибка при обработке платформы $platform: ${ex.message}" }
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
}
