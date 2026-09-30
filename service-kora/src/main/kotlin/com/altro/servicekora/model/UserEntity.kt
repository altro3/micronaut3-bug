package com.altro.servicekora.model

import io.koraframework.database.jdbc.annotation.EntityJdbc

@EntityJdbc
data class UserEntity(
    val id: Long,
    val name: String,
    val email: String,
)
