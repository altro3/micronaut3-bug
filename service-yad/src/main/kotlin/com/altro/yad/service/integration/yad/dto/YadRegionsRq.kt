package com.altro.yad.service.integration.yad.dto

data class YadRegionsRq(
    val method: String = "get",
    val params: Params
) {

    data class Params(
        val DictionaryNames: List<String> = listOf("GeoRegions"),
    )
}
