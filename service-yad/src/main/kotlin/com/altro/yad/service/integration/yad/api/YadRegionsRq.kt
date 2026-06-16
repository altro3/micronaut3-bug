package com.altro.yad.service.integration.yad.api

data class YadRegionsRq(
    val method: String = "get",
    val params: Params
) {

    data class Params(
        val DictionaryNames: List<String> = listOf("GeoRegions"),
    )
}
