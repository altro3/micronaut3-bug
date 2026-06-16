package com.altro.yad.service.integration.yad.dto

data class YadRegionsRs(
    val result: Result?,
) {

    data class Result(
        val GeoRegions: List<GeoRegionItem>?,
    )

    data class GeoRegionItem(
        val RegionId: Long,
        val RegionName: String,
    )
}
