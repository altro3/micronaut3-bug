package com.altro.servicekora.api.dto

import com.alibaba.fastjson2.annotation.JSONCompiled
import io.koraframework.json.common.annotation.Json
import java.math.BigDecimal

@JSONCompiled
@Json
data class UserResponse(
    val id: Long,
    val name: String,
    val email: String,
    val age: Int? = null,
    val balance: BigDecimal? = null,
    val isActive: Boolean? = null,
    val profile: UserProfileResponse? = null,
    val addresses: List<UserAddressResponse>? = null,
    val permissions: List<UserPermissionResponse>? = null,
    val metadata: Map<String, String>? = null,
) {

    @JSONCompiled
    @Json
    data class UserProfileResponse(
        val firstName: String,
        val lastName: String,
        val phoneNumber: String? = null,
        val telegramId: Long? = null,
        val preferedLanguage: String
    )

    @JSONCompiled
    @Json
    data class UserAddressResponse(
        val country: String,
        val city: String,
        val street: String,
        val zipCode: String,
        val isPrimary: Boolean
    )

    @JSONCompiled
    @Json
    data class UserPermissionResponse(
        val role: String,
        val resource: String,
        val canRead: Boolean,
        val canWrite: Boolean,
        val canDelete: Boolean
    )
}
