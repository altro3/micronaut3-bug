package com.altro.yad.service

import com.altro.yad.api.dto.RegionItemDto
import com.altro.yad.model.Campaign
import org.springframework.stereotype.Service

@Service
class InternalApiService(
    private val campaignService: CampaignService,
    private val dictService: DictService,
) {

    fun getCachedRegions(): List<RegionItemDto> =
        dictService.getAvailableRegions().map {
            RegionItemDto(id = it.id, name = it.name)
        }

    fun getCampaign(id: Long): Campaign =
        campaignService.getCampaignById(id)

    /**
     * Инициация Шага 1: Создание локального драфта по имени
     */
    fun startNewCampaign(name: String): Campaign = 
        campaignService.createDraft(name)

    /**
     * Инициация Шага 2: Привязка списка идентификаторов ГЕО-регионов
     */
    fun setCampaignRegions(id: Long, regionIds: List<String>): Campaign = 
        campaignService.bindRegions(id, regionIds)

    /**
     * Инициация Шага 3: Привязка списка идентификаторов возрастных ограничений
     */
    fun setCampaignAges(id: Long, ageIds: List<String>): Campaign = 
        campaignService.bindAges(id, ageIds)

    /**
     * Инициация Шага 4: Добавление текста, автоматический запуск валидации и отправка в API Яндекса
     */
    fun submitCreative(id: Long, text: String): Campaign = 
        campaignService.addCreativeAndValidate(id, text)
}
