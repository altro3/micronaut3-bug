package com.altro.servicekora.api.dto

import com.alibaba.fastjson2.annotation.JSONCompiled
import io.koraframework.json.common.annotation.Json
import io.koraframework.validation.common.annotation.NotBlank
import io.koraframework.validation.common.annotation.NotEmpty
import io.koraframework.validation.common.annotation.Pattern
import io.koraframework.validation.common.annotation.Size
import io.koraframework.validation.common.annotation.Valid
import java.math.BigDecimal

@JSONCompiled
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

    @JSONCompiled
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

    @JSONCompiled
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

    @JSONCompiled
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