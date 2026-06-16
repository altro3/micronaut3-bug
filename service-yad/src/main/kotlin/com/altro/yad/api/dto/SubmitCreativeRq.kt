package com.altro.yad.api.dto

import jakarta.validation.constraints.NotEmpty

data class SubmitCreativeRq(
    @field:NotEmpty
    val text: String,
)
