package com.altro.yad.service

import com.altro.yad.api.dto.RegionItemDto
import com.altro.yad.api.dto.SaveAndPublishCampaignRq
import com.altro.yad.model.Campaign
import org.springframework.stereotype.Service

@Service
class InternalApiService(
    private val campaignService: CampaignService,
    private val dictService: DictService,
) {

    fun getCachedRegions(query: String): List<RegionItemDto> =
        dictService.searchRegionsByQuery(query).map {
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
     * Инициация Шага 4: Только сохранение текста рекламного креатива в JSONB.
     * Кампания переходит в статус CREATIVE_ADDED, но в сеть Яндекса пока ничего не отправляется.
     */
    fun saveCampaignCreative(id: Long, text: String): Campaign =
        campaignService.addCreativeText(id, text)

    /**
     * Инициация Шага 5: Финальная публикация собранной кампании.
     * Запускает локальную валидацию лимитов и делает физический вызов к API Яндекс.Директ.
     */
    fun publishCampaignToYandex(id: Long): Campaign =
        campaignService.validateAndPublishToExternalNetwork(id)

    fun createAndPublishCampaign(rq: SaveAndPublishCampaignRq): Campaign =
        campaignService.createAndPublishCampaign(rq)

}
