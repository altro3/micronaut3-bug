package com.altro.servicekora.api

import com.altro.servicekora.api.dto.CreateUserRequest
import com.altro.servicekora.api.dto.UserResponse
import com.altro.servicekora.service.UserMapper
import com.altro.servicekora.service.UserService
import io.koraframework.http.common.HttpMethod
import io.koraframework.http.common.annotation.HttpRoute
import io.koraframework.http.common.annotation.Path
import io.koraframework.http.server.common.annotation.HttpController
import io.koraframework.validation.common.Validator

@HttpController
class UserController(
    private val userService: UserService,
    private val userMapper: UserMapper,
    private val createUserValidator: Validator<CreateUserRequest>,
) {

    @HttpRoute(method = HttpMethod.GET, path = "/users/{id}")
    fun getUser(@Path id: Long): UserResponse {
        val entity = userService.getUser(id) ?: throw RuntimeException("User not found")
        return userMapper.toResponse(entity)
    }

    @HttpRoute(method = HttpMethod.POST, path = "/users")
    fun createUser(body: CreateUserRequest): UserResponse {
        createUserValidator.validateAndThrow(body)
        val entity = userService.createUser(body.name, body.email)
        return userMapper.toResponse(entity)
    }
}
