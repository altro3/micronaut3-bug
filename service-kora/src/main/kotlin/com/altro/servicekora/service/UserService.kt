package com.altro.servicekora.service

import com.altro.servicekora.model.*
import io.github.oshai.kotlinlogging.*
import io.koraframework.common.annotation.*

@Component
open class UserService(
//    private val userRepository: UserRepository,
) {

    private val log = KotlinLogging.logger {}

//    @Log
//    @Cacheable(UserCache::class)
//    @Retryable(PostgresRetry::class)
//    @Timed(value = "user.service.get", description = "Время получения пользователя")
    open fun getUser(id: Long): UserEntity? {
        log.info { "Запрос мимо кэша! Идем в Postgres за UserEntity с id: $id" }
        return null
//        return userRepository.findById(id)
    }

//    @Log.`in`
//    @Log.out
    open fun createUser(name: String, email: String): UserEntity {
        log.info { "Сохраняем новую сущность в БД для пользователя: $name" }
        return UserEntity(1, name, email)
//        return userRepository.save(name, email)
    }
}
