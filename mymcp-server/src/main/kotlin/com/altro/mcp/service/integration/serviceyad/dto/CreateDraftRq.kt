package com.altro.mcp.service.integration.serviceyad.dto

import jakarta.validation.constraints.NotEmpty

data class CreateDraftRq(
    @field:NotEmpty
    val name: String,
)
