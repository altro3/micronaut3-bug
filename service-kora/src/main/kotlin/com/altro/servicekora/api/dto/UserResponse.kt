package com.altro.servicekora.api.dto

import io.koraframework.json.common.annotation.Json

@Json
data class UserResponse(
    val id: Long,
    val name: String,
    val email: String,
)
