package com.micronaut.bug.config.log

import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.core.Ordered

@ConfigurationProperties("app.log")
class LogProperties(
    var enabledControllerLogging: Boolean = true,
    var order: Int = Ordered.HIGHEST_PRECEDENCE,
    var skipActuator: Boolean = true,
    /**
     * Максимально допустимый размер тела сообщения в логах (в байтах).
     * Значения 0 или -1 отключают ограничение (логируется весь объем).
     */
    @field:Min(-1)
    var limitLogSize: Int = 8192,
    /**
     * Количество байт, которые будут взяты с начала и с конца сообщения
     * при выполнении обрезки (truncate) длинного лога.
     */
    @field:Positive
    var truncateChunkSize: Int = 4096,
    /**
     * Включает форматирование вывода (Pretty Print для JSON, разбор Multipart).
     * Если false — данные выводятся как есть (сырая строка).
     */
    @field:NotNull
    var prettyPrint: Boolean = true,
    var mdcKeys: List<String> = emptyList(),
)
