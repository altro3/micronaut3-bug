package com.micronaut.bug.log.config

import jakarta.validation.Valid
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.core.Ordered
import org.springframework.http.HttpHeaders
import org.springframework.util.unit.DataSize
import java.net.URI
import java.time.Duration

/**
 * Настройки логирования входящих (контроллеры) и исходящих (клиенты) HTTP-запросов.
 * Позволяют гибко управлять детализацией, безопасностью памяти и форматом вывода.
 */
@ConfigurationProperties("app.log")
class LogProperties(
    /**
     * Включает или выключает логирование на уровне контроллеров (ServerLoggingFilter).
     */
    var enabled: Boolean = true,
    /**
     * Порядок выполнения фильтра в цепочке Spring Security / Servlet Filter.
     * По умолчанию имеет наивысший приоритет.
     */
    var order: Int = Ordered.HIGHEST_PRECEDENCE + 1,
    /**
     * Если true, запросы к Spring Boot Actuator (/actuator) не будут логироваться.
     */
    var skipActuator: Boolean = true,
    /**
     * Максимальный размер тела, который мы готовы прочитать в память.
     * Если тело больше — кэширование не производится, логируется заглушка.
     */
    @field:Positive
    var maxPayloadSize: DataSize = DataSize.ofMegabytes(15),
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
    /**
     * Настройки интеграции с VictoriaLogs / OpenTelemetry.
     */
    @field:Valid
    var otlp: OtlpProperties = OtlpProperties(),
    /**
     * Настройки маскирования чувствительных данных.
     */
    var masking: MaskingProperties = MaskingProperties()
) {

    /**
     * Настройки интеграции по протоколу OpenTelemetry (OTLP).
     */
    class OtlpProperties(
        /**
         * Включает или выключает отправку логов в формате OTLP.
         */
        var enabled: Boolean = true,
        /**
         * Эндпоинт сервера для приема OTLP логов (например, VictoriaLogs или OTEL Collector).
         * Стандартный путь обычно заканчивается на /v1/logs.
         */
        var url: URI = URI.create("http://localhost:9428/opentelemetry/v1/logs"),
        /**
         * Размер батча (количество логов) перед отправкой.
         */
        @field:Positive
        var batchSize: Int = 200,
        /**
         * Интервал времени, по истечении которого неполный батч будет отправлен.
         */
        var batchTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Максимальный размер очереди в памяти.
         * Защищает приложение от OOM при всплесках логирования.
         */
        @field:Positive
        var queueCapacity: Int = 10000,
        /**
         * Таймаут ожидания завершения работы воркера при выключении приложения.
         * За это время воркер попытается отправить остатки логов из очереди.
         */
        var stopAwaitTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Таймаут на установку соединения с сервером.
         */
        var connectionTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Таймаут на выполнение HTTP-запроса на пуш логов.
         */
        var requestTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Использовать ли GZIP-сжатие при отправке данных (существенно экономит трафик).
         */
        var useGzip: Boolean = true
    )

    class MaskingProperties(
        /**
         * Включает маскирование.
         */
        var enabled: Boolean = true,
        /**
         * Заголовки, требующие строгой маскировки (Credentials/Sessions).
         */
        var sensitiveHeaders: Set<String> = setOf(
            HttpHeaders.AUTHORIZATION,
            HttpHeaders.PROXY_AUTHORIZATION,
            HttpHeaders.COOKIE,
            HttpHeaders.SET_COOKIE,
        ),
        /**
         * Заголовки, требующие частичной маскировки (Косвенные ПДн).
         */
        var indirectHeaders: Set<String> = setOf(
            HttpHeaders.REFERER,
            HttpHeaders.USER_AGENT,
            "x-forwarded-for",
        ),
        /**
         * Поля для полного затирания (пароли, токены).
         */
        var full: Set<String> = setOf(
            "password",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "api_key",
            "authorization",
            "cvv",
            "cvc",
            "pin",
            "fingerprint",
        ),
        /**
         * Поля для частичного маскирования (ПДн согласно ФЗ-152).
         */
        var partial: Set<String> = setOf(
            // Общие данные и документы
            "fio", "fullname", "firstname", "lastname", "middlename",
            "birth", "passport", "document", "snils", "inn",

            // Контакты и адреса
            "phone", "email", "mail", "address", "location", "city", "street",

            // Профессия и образование
            "education", "profession", "job", "salary", "workplace",
        ),
        /**
         * Символ маски.
         */
        var maskChar: String = "*",
        /**
         * Список префиксов пакетов, для которых включена маскировка заголовков.
         * Если список пуст — маскировка применяется везде.
         */
        val maskedPackages: Set<String> = setOf(
            "com.micronaut.bug",
        ),
    )
}
