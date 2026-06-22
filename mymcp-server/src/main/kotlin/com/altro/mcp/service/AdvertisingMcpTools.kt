package com.altro.mcp.service

import com.altro.mcp.service.integration.serviceyad.ServiceYadClient
import com.altro.mcp.service.integration.serviceyad.dto.BindAgesRq
import com.altro.mcp.service.integration.serviceyad.dto.BindRegionsRq
import com.altro.mcp.service.integration.serviceyad.dto.CampaignStepRs
import com.altro.mcp.service.integration.serviceyad.dto.CreateDraftRq
import com.altro.mcp.service.integration.serviceyad.dto.RegionItemDto
import com.altro.mcp.service.integration.serviceyad.dto.SubmitCreativeRq
import org.springframework.ai.tool.annotation.Tool
import org.springframework.ai.tool.annotation.ToolParam
import org.springframework.stereotype.Component

@Component
class AdvertisingMcpTools(
    private val serviceYadClient: ServiceYadClient,
) {

    @Tool(description = "Создать новый черновик рекламной кампании в Яндекс.Директ. Это СТРОГО первый шаг автоматизации. Вызывай, как только пользователь выразил намерение настроить Яндекс.")
    fun initYandexDraft(
        @ToolParam(description = "Название кампании на русском языке. Если пользователь не назвал его, придумай сам на основе ниши его бизнеса.")
        name: String,
    ): CampaignStepRs? {
        return serviceYadClient.createDraft(CreateDraftRq(name = name))
    }

    @Tool(description = """
        Поиск внутренних числовых ID географических регионов по их текстовому названию (например, 'Москва', 'Новосибирск'). 
        Обязательно вызывай этот инструмент ПЕРЕД bindYandexRegions.
    """)
    fun searchYandexRegions(
        @ToolParam(description = "Уникальный числовой идентификатор текущей кампании (campaignId).")
        campaignId: Long, // <--- ДОБАВЛЯЕМ ОБЯЗАТЕЛЬНЫЙ АРГУМЕНТ ДЛЯ ИИ!

        @ToolParam(description = "Текст запроса. Минимум 3 символа названия города (например, 'новос').")
        query: String
    ): String {
        if (query.trim().length < 3) return "Ошибка: Название города слишком короткое."

        val regions = serviceYadClient.getRegions(query) ?: emptyList()
        if (regions.isEmpty()) return "Регион с названием '$query' не найден."

        val formattedList = regions.mapIndexed { index, dto ->
            "${index + 1}. Название: [${dto.name}], ID ДЛЯ regionIds = '${dto.id}'"
        }.joinToString(separator = "\n")

        return """
            Результаты поиска по запросу '$query':
            $formattedList
            
            ⚠️ ИНСТРУКЦИЯ ДЛЯ АГЕНТА: 
            Выбери подходящий ID региона из списка выше (например, '3'). 
            Затем немедленно вызови инструмент 'bindYandexRegions', передав туда СТРОГО СЛЕДУЮЩИЕ АРГУМЕНТЫ:
            - campaignId: $campaignId (Используй СТРОГО это число! Тебе категорически запрещено его менять или выдумывать другое!)
            - regionIds: [идентификатор выбранного региона]
        """.trimIndent()
    }

    @Tool(description = """
        Привязать выбранные географические регионы к созданной кампании Яндекс.Директ. 
        ⚠️ КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО придумывать ID регионов от себя! 
        Передавай в поле regionIds только те строковые ID, которые ты СТРОГО ТОЛЬКО ЧТО получил из ответа инструмента 'searchYandexRegions'.
    """)
    fun bindYandexRegions(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long,
        @ToolParam(description = "Список строковых ID регионов, полученных строго из searchYandexRegions.") regionIds: List<String>
    ): CampaignStepRs? {
        return serviceYadClient.bindRegions(campaignId, BindRegionsRq(regionIds = regionIds))
    }

    @Tool(description = """
        Привязать возрастные ограничения целевой аудитории (возрастной ценз) к созданной кампании Яндекс.Директ. 
        🔥 ИНСТРУКЦИЯ ДЛЯ АГЕНТА: Как только пользователь явно или косвенно указывает возраст (например: "для взрослых", "не для детей", "маркировка 18+", "0+"), ты ОБЯЗАН НЕМЕДЛЕННО запустить этот инструмент! 
        Ты не имеешь права просто написать текстовый ответ в чат. Ты обязан сгенерировать вызов функции 'bindYandexAges'. 
        В параметр campaignId передавай строго текущий ID кампании. В ageIds передавай массив строковых меток (например, для 'не для детей' или 'для взрослых' передай ["18_PLUS"]).
    """)
    fun bindYandexAges(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long,
        @ToolParam(description = "Список строковых ID возрастных меток (например: '0_PLUS', '12_PLUS', '16_PLUS', '18_PLUS').") ageIds: List<String>
    ): CampaignStepRs? {
        return serviceYadClient.bindAges(campaignId, BindAgesRq(ageIds = ageIds))
    }

    @Tool(description = """
    Сохранить рекламный текст объявления в черновик кампании Яндекс.Директ. 
    🔥 ВАЖНО ДЛЯ АГЕНТА: Этот инструмент является ключевым для настройки кампании. 
    Если у тебя еще нет текста объявления от пользователя, ты имеешь право в любой момент диалога спросить его: 'Какой текст рекламы мы напишем?' 
    или предложить свой готовый продающий вариант на основе его ниши бизнеса, чтобы как можно скорее вызвать этот инструмент.
""")
    fun submitYandexCreative(
        @ToolParam(description = "Уникальный числовой идентификатор кампании.") campaignId: Long,
        @ToolParam(description = "Готовый продающий текст объявления.") text: String
    ): CampaignStepRs? {
        return serviceYadClient.submitCreative(campaignId, SubmitCreativeRq(text = text))
    }

    @Tool(description = """
    Опубликовать готовую рекламную кампанию и отправить её на модерацию в сеть Яндекс.Директ. 
    🔥 ИНСТРУКЦИЯ ДЛЯ АГЕНТА: Этот инструмент вызывается ТОЛЬКО когда кампания полностью собрана (есть гео, возраст, ИНН и текст) И пользователь дал команду на запуск.
    Критерием согласия пользователя являются слова: 'давай', 'запускай', 'публикуй', 'отправляй', 'ок', 'согласен', 'GO'. 
    Как только пользователь написал любое из этих слов в ответ на твой финальный отчет — ты ОБЯЗАН немедленно сгенерировать вызов функции 'publishYandexCampaign'. Не пиши текст, сначала вызови инструмент!
""")
    fun publishYandexCampaign(
        @ToolParam(description = "Уникальный числовой идентификатор кампании (campaignId).") campaignId: Long
    ): CampaignStepRs? {
        return serviceYadClient.publishCampaign(campaignId)
    }

    @Tool(description = "Создать новую рекламную кампанию на платформе ВКонтакте (VK Ads).")
    fun createVkCampaign(
        @ToolParam(description = "Название рекламной кампании.") name: String,
        @ToolParam(description = "Короткий заголовок объявления.") title: String,
        @ToolParam(description = "Основной рекламный текст для баннера.") text: String
    ): String {
        return "Успешно. Кампания в VK Ads инициирована (Холостой режим)."
    }
}
