package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.github.benmanes.caffeine.cache.Cache
import com.github.benmanes.caffeine.cache.Caffeine
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest
import org.springframework.stereotype.Service
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

@Service
class McpDictionaryService(
    private val mcpClient: McpSyncClient,
    private val jsonMapper: JsonMapper
) {
    private val log = KotlinLogging.logger {}

    private val regionsCache: Cache<Platform, List<McpDictionaryItem>> = Caffeine.newBuilder()
        .maximumSize(Platform.entries.size.toLong())
        .expireAfterWrite(1, TimeUnit.HOURS)
        .build()

    private val agesCache: Cache<Platform, List<McpDictionaryItem>> = Caffeine.newBuilder()
        .maximumSize(Platform.entries.size.toLong())
        .expireAfterWrite(1, TimeUnit.HOURS)
        .build()

    /**
     * Получить список регионов. Если данных нет в Caffeine — идем в MCP, иначе — отдаем из памяти.
     */
    fun getRegions(platform: Platform): List<McpDictionaryItem> {
        return regionsCache.get(platform) { key ->
            fetchDictionaryFromMcp(key, "getRegionsList")
        }
    }

    fun getAgeOptions(platform: Platform): List<McpDictionaryItem> {
        return agesCache.get(platform) { key ->
            fetchDictionaryFromMcp(key, "getAgeTargetingOptions")
        }
    }

    /**
     * Принудительный сброс кэша (например, если в рекламном кабинете обновились гео)
     */
    fun invalidateAllCaches(platform: Platform) {
        regionsCache.invalidate(platform)
        agesCache.invalidate(platform)
        log.info { "♻️ Нативные Caffeine-кэши регионов и возрастов для $platform полностью очищены" }
    }

    private fun fetchDictionaryFromMcp(platform: Platform, toolName: String): List<McpDictionaryItem> = try {
        log.info { "📡 КЭШ ПРОМАХ! Запрашиваю инструмент '$toolName' из MCP для платформы: $platform" }

        val request = CallToolRequest(toolName, mapOf("platform" to platform.name), mapOf())
        val response = mcpClient.callTool(request)

        jsonMapper.readValue(response.content.toString(), jacksonTypeRef<List<McpDictionaryItem>>())
    } catch (e: Exception) {
        log.error(e) { "Не удалось подгрузить справочник через инструмент $toolName для платформы $platform" }
        // Безопасный фолбэк, чтобы не валить систему при временном падении MCP-сервера
        getDefaultFallback(toolName)
    }

    private fun getDefaultFallback(toolName: String): List<McpDictionaryItem> {
        return if (toolName == "getRegionsList") {
            listOf(McpDictionaryItem("default_geo", "Россия (Общий)"))
        } else {
            listOf(McpDictionaryItem("0", "Без ограничений (0+)"), McpDictionaryItem("18", "Только взрослые (18+)"))
        }
    }
    data class McpDictionaryItem(
        val id: String,
        val name: String,
    )
}
