package com.altro.yad.service

import com.altro.yad.service.integration.yad.YadClient
import com.altro.yad.service.integration.yad.dto.YadRegionsRq
import com.github.benmanes.caffeine.cache.Cache
import com.github.benmanes.caffeine.cache.Caffeine
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Service
import java.util.concurrent.TimeUnit

@Service
class DictService(
    private val yadClient: YadClient,
) {

    private val log = KotlinLogging.logger {}

    private val regionsCache: Cache<String, List<DictionaryItem>> = Caffeine.newBuilder()
        .maximumSize(1)
        .expireAfterWrite(12, TimeUnit.HOURS)
        .build()

    @Async
    @EventListener(ApplicationReadyEvent::class)
    fun coldStartWarmup() {
        log.info { "🚀 [Cold Start] Запуск авто-прогрева кэша регионов из API Яндекс.Директ..." }
        try {
            getAvailableRegions()
            log.info { "🎯 [Cold Start] Кэш регионов успешно заполнен при старте!" }
        } catch (e: Exception) {
            log.error(e) { "⚠️ [Cold Start] Не удалось прогреть кэш при старте приложения. Будет выполнен ленивый перезапрос." }
        }
    }

    fun getAvailableRegions(): List<DictionaryItem> {
        return regionsCache.get("ALL_REGIONS") { _ ->
            fetchRegionsFromYandex()
        }
    }

    /**
     * Умный поиск регионов для ИИ-оркестратора (через MCP инструмент).
     * Фильтрует локальный кэш Caffeine без обращений к внешнему API.
     */
    fun searchRegionsByQuery(query: String?, maxResults: Int = 10): List<DictionaryItem> {
        // Защита от пустых запросов и слишком коротких фраз (минимум 3 символа)
        if (query == null || query.trim().length < 3) {
            return emptyList()
        }

        val cleanedQuery = query.trim().lowercase()
        val allRegions = getAvailableRegions()

        return allRegions
            .asSequence()
            // Ищем вхождение строки без учета регистра
            .filter { it.name.lowercase().contains(cleanedQuery) }
            // Сортируем: точное совпадение или начало строки идет выше, чем вхождение в середине
            .sortedByDescending {
                val lowerName = it.name.lowercase()
                when {
                    lowerName == cleanedQuery -> 3
                    lowerName.startsWith(cleanedQuery) -> 2
                    else -> 1
                }
            }
            // Ограничиваем размер JSON-ответа для экономии контекста (токенов) LLM
            .take(maxResults)
            .toList()
    }

    private fun fetchRegionsFromYandex(): List<DictionaryItem> {
        log.info { "📡 КЭШ ПРОМАХ/ОБНОВЛЕНИЕ: Физический вызов API Яндекс Dictionaries..." }
        val response = yadClient.getRegions(YadRegionsRq(params = YadRegionsRq.Params()))

        return response?.result?.GeoRegions?.map {
            DictionaryItem(id = it.RegionId.toString(), name = it.RegionName)
        } ?: throw IllegalStateException("API Яндекса вернул пустой справочник регионов")
    }

    data class DictionaryItem(
        val id: String,
        val name: String,
    )
}
