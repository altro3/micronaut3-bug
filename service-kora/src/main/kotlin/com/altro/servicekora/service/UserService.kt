package com.altro.servicekora.service

import com.altro.servicekora.model.UserEntity
import com.altro.servicekora.repository.UserRepository
import io.github.oshai.kotlinlogging.KotlinLogging
import io.koraframework.cache.annotation.Cacheable
import io.koraframework.common.annotation.Component
import io.koraframework.logging.common.annotation.Log
import io.koraframework.resilient.retry.annotation.Retryable
import io.micrometer.core.annotation.Timed

@Component
open class UserService(
    private val userRepository: UserRepository,
) {

    private val log = KotlinLogging.logger {}

    @Log
//    @Cacheable(UserCache::class)
    @Retryable(PostgresRetry::class)
    @Timed(value = "user.service.get", description = "Время получения пользователя")
    open fun getUser(id: Long): UserEntity? {
        log.info { "Запрос мимо кэша! Идем в Postgres за UserEntity с id: $id" }
        return userRepository.findById(id)
    }

    @Log.`in`
    @Log.out
    open fun createUser(name: String, email: String): UserEntity {
        log.info { "Сохраняем новую сущность в БД для пользователя: $name" }
        return userRepository.save(name, email)
    }
}
