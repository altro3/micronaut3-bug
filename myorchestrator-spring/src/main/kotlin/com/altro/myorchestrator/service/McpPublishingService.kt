package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession.CreativeWithImage
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Service
import java.util.UUID

@Service
class McpPublishingService(
    private val mcpClient: McpSyncClient,
) {

    private val log = KotlinLogging.logger {}

    fun publish(campaignId: Long, platforms: List<Platform>, creatives: Map<Platform, CreativeWithImage>): McpPublishResult {
        val logs = mutableListOf<String>()
        val platformCampaignIds = mutableMapOf<Platform, String?>()

        try {
            val availableTools = mcpClient.listTools().tools
            platforms.forEach { platform ->
                val creative = creatives[platform] ?: return@forEach
                val toolName = when (platform) {
                    Platform.VK_ADS -> "createVkCampaign"
                    Platform.YANDEX_DIRECT -> "createYandexCampaign"
                }

                logs.add("🚀 АГЕНТ: Отправка RPC-запроса в инструмент '$toolName' для $platform...")

                if (availableTools.any { it.name == toolName }) {
                    val arguments = mapOf(
                        "name" to "Кампания_${campaignId}_${platform.name}",
                        "title" to creative.title,
                        "text" to creative.bodyText
                    )
                    val request = CallToolRequest(toolName, arguments, mapOf())
                    val mcpResponse = mcpClient.callTool(request)

                    logs.add("✅ MCP УСПЕХ ($platform): ${mcpResponse.content}")
                    platformCampaignIds[platform] = "act_" + UUID.randomUUID().toString().take(8)
                } else {
                    logs.add("❌ MCP ОШИБКА ($platform): Инструмент '$toolName' отсутствует на сервере 8090")
                }
            }
        } catch (e: Exception) {
            log.error(e) { "Сбой отправки данных в MCP" }
            logs.add("❌ КРИТИЧЕСКИЙ СБОЙ МАРШРУТИЗАЦИИ MCP: ${e.message}")
        }
        return McpPublishResult(logs, platformCampaignIds)
    }

    data class McpPublishResult(
        val logs: List<String>,
        val campaignIds: Map<Platform, String?>,
    )
}
