package com.altro.yad.service

import com.altro.yad.model.Campaign
import com.altro.yad.model.CampaignStatus
import com.altro.yad.repository.CampaignRepository
import com.altro.yad.service.integration.yad.YadClient
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRq // АКТУАЛЬНЫЙ ПАКЕТ DTO
import com.altro.yad.service.integration.yad.dto.YadCreateCampaignRq.YadCampaignItem // АКТУАЛЬНЫЙ ПАКЕТ DTO
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
     * ШАГ 4: Добавление текста креатива и запуск локальной проверки лимитов
     */
    @Transactional
    fun addCreativeAndValidate(id: Long, text: String): Campaign {
        val campaign = campaignRepository.findById(id)
            .orElseThrow { IllegalArgumentException("Кампания с ID $id не найдена") }

        campaign.data.text = text
        campaign.status = CampaignStatus.CREATIVE_ADDED
        campaign.updatedAt = Instant.now()

        val saved = campaignRepository.save(campaign)
        log.info { "💾 [ЯД-Адаптер] Шаг 4: Добавлен текст креатива для ID ${saved.id}. Запуск валидации..." }

        return validateAndSync(saved)
    }

    /**
     * ШАГ 5: Внутренняя валидация лимитов Яндекса (не более 81 символа)
     */
    private fun validateAndSync(campaign: Campaign): Campaign {
        val campaignText = campaign.data.text ?: ""

        if (campaignText.length > 81) {
            campaign.status = CampaignStatus.SYNC_ERROR
            campaign.errorMessage = "Локальная валидация провалена: Текст объявления длиннее 81 символа"
            campaign.updatedAt = Instant.now()
            return campaignRepository.save(campaign).also {
                log.warn { "❌ [ЯД-Адаптер] ID ${it.id} не прошел лимиты символов" }
            }
        }

        campaign.status = CampaignStatus.VALIDATED
        campaign.updatedAt = Instant.now()
        val validated = campaignRepository.save(campaign)
        log.info { "💾 [ЯД-Адаптер] Шаг 5: Локальная валидация ID ${validated.id} пройдена. Уходим в сеть..." }

        return executeExternalSync(validated)
    }

    /**
     * ШАГ 6: Физический HTTP-вызов через YadClient к API Яндекс.Директ
     */
    private fun executeExternalSync(campaign: Campaign): Campaign {
        campaign.status = CampaignStatus.SYNCING
        campaign.errorMessage = null
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
                syncingCampaign.errorMessage = null
            } else {
                val errorDetails = addResult?.Errors?.joinToString { "${it.Code}: ${it.Message}" } ?: "Ошибка API"
                syncingCampaign.status = CampaignStatus.SYNC_ERROR
                syncingCampaign.errorMessage = errorDetails
            }
            campaignRepository.save(syncingCampaign)
        } catch (e: Exception) {
            syncingCampaign.status = CampaignStatus.SYNC_ERROR
            syncingCampaign.errorMessage = "Сетевой критический сбой: ${e.message}"
            syncingCampaign.updatedAt = Instant.now()
            campaignRepository.save(syncingCampaign)
        }
    }
}
