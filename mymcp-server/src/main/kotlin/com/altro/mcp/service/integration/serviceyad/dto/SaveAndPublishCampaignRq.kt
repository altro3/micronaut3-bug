package com.altro.mcp.service.integration.serviceyad.dto

data class SaveAndPublishCampaignRq(
    val name: String,
    val regionNames: List<String>,
    val ageLimit: String,
    val inn: String,
    val creativeText: String,
    val autoPublish: Boolean,
)
