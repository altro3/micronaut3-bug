package com.micronaut.bug.service

import com.micronaut.bug.config.User
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.stereotype.Service

@Service
class BusinessService {

    private val log = KotlinLogging.logger {}

    fun processOrder(user: User?): User? {
        if (user == null) {
            log.warn { "Пользователь не найден, пропускаем заказ" }
            return null
        }

        // В Kotlin обращение к полям data-класса идет через имя свойства, а не через метод ()
        log.info { "Оформляем заказ для пользователя: ${user.username} (ID: ${user.id})" }

        return user
    }

    fun someFunc() {

        log.info { "Стартуем какую-то функцию" }

        throw IllegalArgumentException("Что-то пошло не так")
    }
}
