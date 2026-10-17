package com.altro.servicekora.api.dto

import io.koraframework.json.common.annotation.*
import java.math.*

@Json
data class CreateUserRequest(
    val name: String,
    val email: String,

    val age: Int? = null,
    val balance: BigDecimal? = null,
    val isActive: Boolean,
    val profile: UserProfile? = null,
    val addresses: List<UserAddress>? = null,
    val permissions: List<UserPermission>? = null,
    val metadata: Map<String, String>?
) {

    @Json
    data class UserProfile(
        val firstName: String,
        val lastName: String,
        val phoneNumber: String? = null,
        val telegramId: Long? = null,
        val preferedLanguage: String
    )

    @Json
    data class UserAddress(
        val country: String,
        val city: String,
        val street: String,
        val zipCode: String,
        val isPrimary: Boolean
    )

    @Json
    data class UserPermission(
        val role: String,
        val resource: String,
        val canRead: Boolean,
        val canWrite: Boolean,
        val canDelete: Boolean
    )
}