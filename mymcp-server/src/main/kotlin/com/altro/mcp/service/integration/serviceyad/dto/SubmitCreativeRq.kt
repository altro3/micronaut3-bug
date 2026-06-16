package com.altro.mcp.service.integration.serviceyad.dto

import jakarta.validation.constraints.NotEmpty

data class SubmitCreativeRq(
    @field:NotEmpty
    val text: String,
)
