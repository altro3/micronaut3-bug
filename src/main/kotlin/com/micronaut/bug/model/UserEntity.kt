package com.micronaut.bug.model

import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table
import java.time.Instant

@Table("users")
data class UserEntity(
    @Id
    val id: Long? = null,
    val username: String,
    val email: String,
    val isActive: Boolean,
    val createdAt: Instant,
)
