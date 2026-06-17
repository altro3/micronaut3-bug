package com.altro.yad.service

import com.altro.yad.model.Campaign
import com.altro.yad.model.CampaignStatus
import com.altro.yad.repository.CampaignRepository
import com.altro.yad.service.integration.yad.YadClient
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRq
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRq.YadCampaignItem
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Instant

@Service
class CampaignService(
    private val campaignRepository: CampaignRepository,
    private val yadClient: YadClient
) {
    private val log = KotlinLogging.logger {}

    fun getCampaignById(id: Long): Campaign =
        campaignRepository.findByIdOrNull(id) ?: throw IllegalArgumentException("Кампания с ID $id не найдена в адаптере Яндекса")

    /**
     * ШАГ 1: Создание стартового черновика (Имя)
     */
    @Transactional
    fun createDraft(name: String): Campaign {
        val campaign = Campaign(
            createdAt = Instant.now(),
            updatedAt = Instant.now()
        ).apply {
            status = CampaignStatus.DRAFT
            data.name = name
        }
        return campaignRepository.save(campaign).also {
            log.info { "💾 [ЯД-Адаптер] Шаг 1: Создан черновик кампании. ID: ${it.id}" }
        }
    }

    /**
     * ШАГ 2: Привязка выбранных ГЕО-регионов
     */
    @Transactional
    fun bindRegions(id: Long, regionIds: List<String>): Campaign {
        val campaign = campaignRepository.findById(id)
            .orElseThrow { IllegalArgumentException("Кампания с ID $id не найдена") }

        campaign.data.targetRegionIds = regionIds
        campaign.status = CampaignStatus.REGIONS_BOUND
        campaign.updatedAt = Instant.now()

        return campaignRepository.save(campaign).also {
            log.info { "💾 [ЯД-Адаптер] Шаг 2: К кампании ID ${it.id} привязаны регионы $regionIds" }
        }
    }

    /**
     * ШАГ 3: Привязка возрастного таргетинга
     */
    @Transactional
    fun bindAges(id: Long, ageIds: List<String>): Campaign {
        val campaign = campaignRepository.findById(id)
            .orElseThrow { IllegalArgumentException("Кампания с ID $id не найдена") }

        campaign.data.targetAgeIds = ageIds
        campaign.status = CampaignStatus.AGES_BOUND
        campaign.updatedAt = Instant.now()

        return campaignRepository.save(campaign).also {
            log.info { "💾 [ЯД-Адаптер] Шаг 3: К кампании ID ${it.id} привязан возраст $ageIds" }
        }
    }

    /**
     * ШАГ 4 (СКОРРЕКТИРОВАН): Только добавление текста креатива в базу данных.
     * Мы БОЛЬШЕ НЕ ВЫЗЫВАЕМ автоматическую синхронизацию с Яндексом.
     */
    @Transactional
    fun addCreativeText(id: Long, text: String): Campaign {
        val campaign = campaignRepository.findById(id)
            .orElseThrow { IllegalArgumentException("Кампания с ID $id не найдена") }

        campaign.data.text = text
        campaign.status = CampaignStatus.CREATIVE_ADDED
        campaign.updatedAt = Instant.now()

        return campaignRepository.save(campaign).also {
            log.info { "💾 [ЯД-Адаптер] Шаг 4: Текст креатива для ID ${it.id} успешно сохранен в JSONB. Кампания ждет публикации." }
        }
    }

    /**
     * ШАГ 5 (НОВЫЙ ПУБЛИЧНЫЙ ТРИГГЕР): Валидация накопленных данных и отправка в сеть Яндекса.
     * Вызывается оркестратором или кнопкой "Запустить рекламу".
     */
    @Transactional
    fun validateAndPublishToExternalNetwork(id: Long): Campaign {
        val campaign = campaignRepository.findById(id)
            .orElseThrow { IllegalArgumentException("Кампания с ID $id не найдена") }

        log.info { "💾 [ЯД-Адаптер] Шаг 5: Инициирован паблишинг для ID ${campaign.id}. Запуск локальной валидации..." }

        // Запускаем вашу цепочку внутренней проверки и физического вызова API
        return validateAndSync(campaign)
    }

    /**
     * Внутренняя валидация лимитов Яндекса (не более 81 символа)
     */
    private fun validateAndSync(campaign: Campaign): Campaign {
        val campaignText = campaign.data.text ?: ""

        if (campaignText.length > 81) {
            campaign.status = CampaignStatus.SYNC_ERROR
            campaign.data.errorMessage = "Локальная валидация провалена: Текст объявления длиннее 81 символа"
            campaign.updatedAt = Instant.now()
            return campaignRepository.save(campaign).also {
                log.warn { "❌ [ЯД-Адаптер] ID ${it.id} не прошел лимиты символов" }
            }
        }

        campaign.status = CampaignStatus.VALIDATED
        campaign.updatedAt = Instant.now()
        val validated = campaignRepository.save(campaign)
        log.info { "💾 [ЯД-Адаптер] Локальная валидация ID ${validated.id} пройдена. Уходим в сеть..." }

        return executeExternalSync(validated)
    }

    /**
     * Физический HTTP-вызов через YadClient к API Яндекс.Директ
     */
    private fun executeExternalSync(campaign: Campaign): Campaign {
        campaign.status = CampaignStatus.SYNCING
        campaign.data.errorMessage = null
        campaign.updatedAt = Instant.now()
        val syncingCampaign = campaignRepository.save(campaign)

        val requestDto = YadCreateCampaignRq(
            params = YadCreateCampaignRq.Params(
                Campaigns = listOf(
                    YadCampaignItem(
                        Name = syncingCampaign.data.name ?: "Без имени",
                        Text = syncingCampaign.data.text ?: "",
                    )
                )
            )
        )

        return try {
            val response = yadClient.createCampaign(requestDto)
            val addResult = response?.result?.AddResults?.firstOrNull()

            syncingCampaign.updatedAt = Instant.now()

            if (addResult?.Id != null) {
                syncingCampaign.externalId = addResult.Id
                syncingCampaign.status = CampaignStatus.SYNCED
                syncingCampaign.data.errorMessage = null
            } else {
                val errorDetails = addResult?.Errors?.joinToString { "${it.Code}: ${it.Message}" } ?: "Ошибка API"
                syncingCampaign.status = CampaignStatus.SYNC_ERROR
                syncingCampaign.data.errorMessage = errorDetails
            }
            campaignRepository.save(syncingCampaign)
        } catch (e: Exception) {
            syncingCampaign.status = CampaignStatus.SYNC_ERROR
            syncingCampaign.data.errorMessage = "Сетевой критический сбой: ${e.message}"
            syncingCampaign.updatedAt = Instant.now()
            campaignRepository.save(syncingCampaign)
        }
    }
}
