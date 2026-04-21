package com.micronaut.bug.config.log

import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.core.Ordered
import org.springframework.util.unit.DataSize

/**
 * Настройки логирования входящих (контроллеры) и исходящих (клиенты) HTTP-запросов.
 * Позволяют гибко управлять детализацией, безопасностью памяти и форматом вывода.
 */
@ConfigurationProperties("app.log")
class LogProperties(
    /**
     * Включает или выключает логирование на уровне контроллеров (ServerLoggingFilter).
     */
    var enabledControllerLogging: Boolean = true,
    /**
     * Порядок выполнения фильтра в цепочке Spring Security / Servlet Filter.
     * По умолчанию имеет наивысший приоритет.
     */
    var order: Int = Ordered.HIGHEST_PRECEDENCE,
    /**
     * Если true, запросы к Spring Boot Actuator (/actuator) не будут логироваться.
     */
    var skipActuator: Boolean = true,
    /**
     * Максимальный размер тела, который мы готовы прочитать в память.
     * Если тело больше — кэширование не производится, логируется заглушка.
     */
    @field:Positive
    var maxPayloadSize: DataSize = DataSize.ofMegabytes(15), // 15MB по умолчанию
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
    /**
     * Список ключей из MDC (Mapped Diagnostic Context), значения которых будут
     * дополнительно выведены в текстовый блок лога.
     * Полезно для визуальной связки лога с конкретным контекстом (например, userId, traceId).
     */
    var mdcKeys: List<String> = emptyList(),
)
