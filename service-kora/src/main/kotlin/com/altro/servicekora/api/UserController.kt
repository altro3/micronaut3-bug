package com.altro.servicekora.api

import com.altro.servicekora.api.dto.CreateUserRequest
import com.altro.servicekora.api.dto.UserResponse
import com.altro.servicekora.service.UserMapper
import com.altro.servicekora.service.UserService
import io.koraframework.common.annotation.Component
import io.koraframework.http.common.HttpMethod
import io.koraframework.http.common.annotation.HttpRoute
import io.koraframework.http.common.annotation.Path
import io.koraframework.http.common.annotation.Query
import io.koraframework.http.server.common.annotation.HttpController
import io.koraframework.json.common.annotation.Json
import io.koraframework.validation.common.Validator
import io.koraframework.validation.common.annotation.NotBlank
import io.koraframework.validation.common.annotation.Size
import io.koraframework.validation.common.annotation.Valid
import io.koraframework.validation.common.annotation.Validate

@HttpController
@Component
@Valid
open class UserController(
    private val userService: UserService,
    private val userMapper: UserMapper,
    private val createUserValidator: Validator<CreateUserRequest>,
) {

    @Json
    @HttpRoute(method = HttpMethod.GET, path = "/users/{id}")
    open fun getUser(@Path id: Long): UserResponse {
        val entity = userService.getUser(id) ?: throw RuntimeException("User not found")
        return userMapper.toResponse(entity)
    }

    @Json
    @HttpRoute(method = HttpMethod.POST, path = "/users")
    open fun createUser(@Json body: CreateUserRequest): UserResponse {
        createUserValidator.validateAndThrow(body)
        val entity = userService.createUser(body.name, body.email)
        return userMapper.toResponse(entity)
    }

    @Validate
    @Json
    @HttpRoute(method = HttpMethod.GET, path = "/users/search")
    open fun searchUser(
        @Query @NotBlank @Size(min = 3, max = 20) query: String
    ): List<UserResponse> {
        return emptyList()
    }
}
