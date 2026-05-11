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

@ConfigurationProperties("app.log")
class LogProperties(
    var enabled: Boolean = true,
    var order: Int = Ordered.HIGHEST_PRECEDENCE + 1,
    var skipActuator: Boolean = true,
    @field:Positive
    var maxPayloadSize: DataSize = DataSize.ofMegabytes(15),
    @field:Min(-1)
    var limitLogSize: Int = 8192,
    @field:Positive
    var truncateChunkSize: Int = 4096,
    var stackTraceMaxLines: Int = 10,
    var stackTraceRootCauseFull: Boolean = true,
    @field:NotNull
    var prettyPrint: Boolean = true,
    var mdcKeys: List<String> = emptyList(),
    @field:Valid
    var otlp: OtlpProperties = OtlpProperties(),
    var masking: MaskingProperties = MaskingProperties(),
) {

    class OtlpProperties(
        var enabled: Boolean = true,
        var url: URI = URI.create("http://localhost:9428/opentelemetry/v1/logs"),
        @field:Positive
        var batchSize: Int = 200,
        var batchTimeout: Duration = Duration.ofSeconds(5),
        @field:Positive
        var queueCapacity: Int = 10000,
        var stopAwaitTimeout: Duration = Duration.ofSeconds(5),
        var connectionTimeout: Duration = Duration.ofSeconds(5),
        var requestTimeout: Duration = Duration.ofSeconds(5),
        var useGzip: Boolean = true,
        @field:Positive
        var compressionThreshold: DataSize = DataSize.ofKilobytes(4),
    )

    class MaskingProperties(
        var enabled: Boolean = true,
        var sensitiveHeaders: Set<String> = setOf(
            HttpHeaders.AUTHORIZATION,
            HttpHeaders.PROXY_AUTHORIZATION,
            HttpHeaders.COOKIE,
            HttpHeaders.SET_COOKIE,
        ),
        var indirectHeaders: Set<String> = setOf(
            HttpHeaders.REFERER,
            HttpHeaders.USER_AGENT,
            "x-forwarded-for",
        ),
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
        var partial: Set<String> = setOf(
            // Общие данные и документы
            "fio", "fullname", "firstname", "lastname", "middlename",
            "birth", "passport", "document", "snils", "inn",

            // Контакты и адреса
            "phone", "email", "mail", "address", "location", "city", "street",

            // Профессия и образование
            "education", "profession", "job", "salary", "workplace",
        ),
        var maskChar: String = "*",
        val maskedPackages: Set<String> = setOf(
            "com.micronaut.bug",
        ),
    )
}
