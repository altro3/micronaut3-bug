package com.altro.yad.service.integration.yad.api

data class YadCreateCampaignRq(
    val method: String = "add",
    val params: Params
) {
    data class Params(val Campaigns: List<YadCampaignItem>)
    data class YadCampaignItem(
        val Name: String,
        val Text: String,
        val TextCampaign: TextCampaignOptions = TextCampaignOptions()
    )
    data class TextCampaignOptions(
        val BiddingStrategy: Strategy = Strategy()
    )
    data class Strategy(
        val Search: SearchOptions = SearchOptions()
    )
    data class SearchOptions(
        val BiddingStrategyType: String = "HIGHEST_POSITION"
    )
}
