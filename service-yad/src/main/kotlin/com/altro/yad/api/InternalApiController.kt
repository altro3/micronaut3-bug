package com.altro.yad.api

import com.altro.yad.api.dto.BindAgesRq
import com.altro.yad.api.dto.BindRegionsRq
import com.altro.yad.api.dto.CreateDraftRq
import com.altro.yad.api.dto.SubmitCreativeRq
import com.altro.yad.model.Campaign
import com.altro.yad.service.InternalApiService
import jakarta.validation.Valid
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class InternalApiController(
    private val internalApiService: InternalApiService,
) {

    @PostMapping("/internal/campaigns/draft")
    fun createDraft(@RequestBody @Valid rq: CreateDraftRq): Campaign =
        internalApiService.startNewCampaign(rq.name)

    @PutMapping("/internal/campaigns/{id}/regions")
    fun bindRegions(
        @PathVariable id: Long,
        @RequestBody @Valid rq: BindRegionsRq,
    ): Campaign =
        internalApiService.setCampaignRegions(id, rq.regionIds)

    @PutMapping("/internal/campaigns/{id}/ages")
    fun bindAges(
        @PathVariable id: Long,
        @RequestBody @Valid rq: BindAgesRq,
    ): Campaign =
        internalApiService.setCampaignAges(id, rq.ageIds)

    @PostMapping("/internal/campaigns/{id}/creative")
    fun submitCreative(
        @PathVariable id: Long,
        @RequestBody @Valid rq: SubmitCreativeRq,
    ): Campaign =
        internalApiService.submitCreative(id, rq.text)
}
