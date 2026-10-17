package com.altro.servicekora.api.dto

import io.koraframework.json.common.annotation.*
import io.koraframework.validation.common.annotation.*
import java.math.*

@Json
@Valid
data class CreateUserRequest(
    @field:Size(min = 2, max = 50)
    val name: String,

    @field:NotEmpty
    @field:Pattern("^[A-Za-z0-9+_.-]+@(.+)$")
    val email: String,

    val age: Int? = null,
    val balance: BigDecimal? = null,
    val isActive: Boolean,

    @field:Valid
    val profile: UserProfile? = null,
    @field:NotEmpty
    @field:Valid
    val addresses: List<UserAddress>? = null,

    @field:Valid
    val permissions: List<UserPermission>? = null,
    val metadata: Map<String, String>?
) {

    @Json
    @Valid
    data class UserProfile(
        @field:NotBlank
        val firstName: String,
        @field:NotBlank
        val lastName: String,
        val phoneNumber: String? = null,
        val telegramId: Long? = null,
        val preferedLanguage: String
    )

    @Json
    @Valid
    data class UserAddress(
        @field:NotBlank
        val country: String,
        @field:NotBlank
        val city: String,
        @field:NotBlank
        val street: String,
        val zipCode: String,
        val isPrimary: Boolean
    )

    @Json
    @Valid
    data class UserPermission(
        val role: String,
        val resource: String,
        val canRead: Boolean,
        val canWrite: Boolean,
        val canDelete: Boolean
    )
}