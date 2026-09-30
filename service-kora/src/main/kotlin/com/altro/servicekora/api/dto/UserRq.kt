package com.altro.servicekora.api.dto

import io.koraframework.json.common.annotation.Json

@Json
data class UserRq(
    val name: String,
    val email: String
)
