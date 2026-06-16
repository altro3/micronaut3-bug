package com.altro.yad.api

import com.altro.yad.api.dto.*
import com.altro.yad.model.Campaign
import com.altro.yad.service.InternalApiService
import jakarta.validation.Valid
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
     * Шаг 4: Добавить рекламный текст креатива, провести валидацию и отправить в сеть Яндекса
     * POST http://localhost:8090/internal/campaigns/{id}/creative
     */
    @PostMapping("/internal/campaigns/{id}/creative")
    fun submitCreative(
        @PathVariable id: Long,
        @RequestBody @Valid rq: SubmitCreativeRq
    ): CampaignStepRs =
        internalApiService.submitCreative(id, rq.text).toRs()

    private fun Campaign.toRs() = CampaignStepRs(
        id = this.id,
        status = this.status,
        externalId = this.externalId,
        errorMessage = this.errorMessage
    )
}
