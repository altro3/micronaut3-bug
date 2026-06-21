package com.altro.yad.api

import com.altro.yad.api.dto.*
import com.altro.yad.model.Campaign
import com.altro.yad.service.InternalApiService
import jakarta.validation.Valid
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import org.springframework.web.bind.annotation.*

@RestController
class InternalCampaignController(
    private val internalApiService: InternalApiService
) {

    /**
     * Шаг 1: Создать черновик кампании (Имя)
     * POST http://localhost:8090/internal/campaigns/draft
     */
    @PostMapping("/internal/campaigns/draft")
    fun createDraft(@RequestBody @Valid rq: CreateDraftRq): CampaignStepRs =
        internalApiService.startNewCampaign(rq.name).toRs()

    /**
     * Шаг 2: Привязать выбранные ГЕО-регионы к кампании
     * PUT http://localhost:8090/internal/campaigns/{id}/regions
     */
    @PutMapping("/internal/campaigns/{id}/regions")
    fun bindRegions(
        @PathVariable id: Long,
        @RequestBody @Valid rq: BindRegionsRq
    ): CampaignStepRs =
        internalApiService.setCampaignRegions(id, rq.regionIds).toRs()

    /**
     * Шаг 3: Привязать возрастной таргетинг к кампании
     * PUT http://localhost:8090/internal/campaigns/{id}/ages
     */
    @PutMapping("/internal/campaigns/{id}/ages")
    fun bindAges(
        @PathVariable id: Long,
        @RequestBody @Valid rq: BindAgesRq
    ): CampaignStepRs =
        internalApiService.setCampaignAges(id, rq.ageIds).toRs()

    /**
     * Шаг 4: Добавить рекламный текст креатива БЕЗ отправки в сеть
     * POST http://localhost:8081/internal/campaigns/{id}/creative
     */
    @PostMapping("/internal/campaigns/{id}/creative")
    fun saveCreative(
        @PathVariable id: Long,
        @RequestBody @Valid rq: SubmitCreativeRq
    ): CampaignStepRs =
        internalApiService.saveCampaignCreative(id, rq.text).toRs()

    /**
     * Шаг 5: Финальный триггер публикации собранной кампании в Яндекс
     * POST http://localhost:8081/internal/campaigns/{id}/publish
     */
    @PostMapping("/internal/campaigns/{id}/publish")
    fun publishCampaign(@PathVariable id: Long): CampaignStepRs =
        internalApiService.publishCampaignToYandex(id).toRs()
    /**
     * Чтение текущего стейта кампании
     * GET http://localhost:8090/internal/campaigns/{id}
     */
    @GetMapping("/internal/campaigns/{id}")
    fun getCampaignStatus(@PathVariable id: Long): CampaignStepRs =
        internalApiService.getCampaign(id).toRs()

    /**
     * Раздача ГЕО-регионов из Caffeine кэша
     * GET http://localhost:8090/internal/dictionaries/regions
     */
    @GetMapping("/internal/dictionaries/regions")
    fun getRegions(@RequestParam @NotBlank @Size(min = 3) query: String): List<RegionItemDto> =
        internalApiService.getCachedRegions(query)

    private fun Campaign.toRs() = CampaignStepRs(
        campaignId = this.id,
        status = this.status,
        externalId = this.externalId,
        errorMessage = this.data.errorMessage
    )
}
