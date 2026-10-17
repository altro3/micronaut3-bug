package com.altro.servicekora.model

import io.koraframework.json.common.annotation.*

@Json
//@EntityJdbc
data class UserEntity(
    val id: Long,
    val name: String,
    val email: String,
)
