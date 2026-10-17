package com.altro.servicekora.api

import com.altro.servicekora.api.dto.*
import com.altro.servicekora.api.dto.UserResponse.*
import com.altro.servicekora.service.*
import io.koraframework.common.annotation.*
import io.koraframework.http.common.*
import io.koraframework.http.common.annotation.*
import io.koraframework.http.server.common.annotation.*
import io.koraframework.json.common.annotation.*
import java.math.*

@HttpController
@Component
open class UserController(
    private val userService: UserService,
    private val userMapper: UserMapper,
) {

    @Json
    @HttpRoute(method = HttpMethod.GET, path = "/users/{id}")
    open fun getUser(@Path id: Long): UserResponse? {
        val entity = userService.getUser(id) ?: throw RuntimeException("User not found")
        return userMapper.toResponse(entity)
    }

    @Json
    @HttpRoute(method = HttpMethod.POST, path = "/users")
    open fun createUser(@Json body: CreateUserRequest): UserResponse? {
        val entity = userService.createUser(body.name, body.email)
        return userMapper.toResponse(entity)
    }

    @HttpRoute(method = HttpMethod.POST, path = "/users/search")
    @Json
    open fun searchUser(
        @Json body: CreateUserRequest,
    ): List<UserResponse> {

        val createHeavyUser = { id: Long, name: String ->
            UserResponse(
                id = id,
                name = name,
                email = "${name.lowercase().replace(" ", "")}@example.com",
                age = 30 + id.toInt(),
                balance = BigDecimal("15000.75").multiply(BigDecimal.valueOf(id)),
                isActive = true,
                profile = UserProfileResponse(
                    firstName = name.split(" ")[0],
                    lastName = name.split(" ")[1],
                    phoneNumber = "+7999123456$id",
                    telegramId = 123456789L + id,
                    preferedLanguage = "RU"
                ),
                addresses = listOf(
                    UserAddressResponse("Russia", "Moscow", "Tverskaya St, $id", "101000", true),
                    UserAddressResponse("Russia", "Novosibirsk", "Nikolaeva St, ${id + 1}", "630090", false)
                ),
                permissions = listOf(
                    UserPermissionResponse("ADMIN", "USERS", true, true, true),
                    UserPermissionResponse("USER", "BILLING", true, false, false)
                ),
                metadata = mapOf(
                    "department" to "Highload Engineering",
                    "environment" to "production",
                    "cluster_node" to "k8s-node-0$id"
                )
            )
        }

        return listOf(
            createHeavyUser(1L, "Ivan Ivanov"),
            createHeavyUser(2L, "Petr Petrov"),
            createHeavyUser(3L, "Sidor Sidorov")
        )
    }
}
