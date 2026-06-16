package com.altro.yad.api.dto

import jakarta.validation.constraints.NotEmpty

data class CreateDraftRq(
    @field:NotEmpty
    val name: String,
)
