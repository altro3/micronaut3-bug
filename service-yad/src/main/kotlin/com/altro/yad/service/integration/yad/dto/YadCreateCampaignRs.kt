package com.altro.yad.service.integration.yad.dto

data class YadCreateCampaignRs(
    val result: Result?,
) {

    data class Result(
        val AddResults: List<AddResultItem>?,
    )

    data class AddResultItem(
        val Id: Long?,
        val Errors: List<YadError>?,
    )

    data class YadError(
        val Code: Int,
        val Message: String,
        val Details: String?,
    )
}
