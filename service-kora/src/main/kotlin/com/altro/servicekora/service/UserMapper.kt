package com.altro.servicekora.service

import com.altro.servicekora.api.dto.UserResponse
import com.altro.servicekora.model.UserEntity
import io.koraframework.common.annotation.Component

@Component
class UserMapper {

    fun toResponse(entity: UserEntity): UserResponse {
        return UserResponse(
            id = entity.id,
            name = entity.name,
            email = entity.email
        )
    }
}
