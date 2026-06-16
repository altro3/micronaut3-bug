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

    // Ключом будет просто статическая строка "ALL_REGIONS"
    private val regionsCache: Cache<String, List<DictionaryItem>> = Caffeine.newBuilder()
        .maximumSize(1)
        .expireAfterWrite(12, TimeUnit.HOURS) // Обновляем раз в 12 часов
        .build()

    /**
     * Автоматический прогрев кэша ГЕО-регионов при старте приложения
     */
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

    /**
     * Быстрая отдача списка регионов из Caffeine
     */
    fun getAvailableRegions(): List<DictionaryItem> {
        return regionsCache.get("ALL_REGIONS") { _ ->
            fetchRegionsFromYandex()
        }
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
