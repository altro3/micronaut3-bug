package com.micronaut.bug.client

import jakarta.validation.Valid
import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import jakarta.validation.constraints.PositiveOrZero
import org.springframework.boot.context.properties.NestedConfigurationProperty
import org.springframework.validation.annotation.Validated
import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_CONNECT_TIMEOUT
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_MAX_ATTEMPTS
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_READ_TIMEOUT
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_DELAY
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_MULTIPLIER
import java.net.URI
import java.time.Duration

@Validated
class HttpClientProperties(
    @field:NotNull
    var url: URI = URI(""),
    var serviceName: String? = null,
    @field:NotNull
    var type: ClientType = INTERNAL,
    var apiKey: String? = null,
    @field:NotNull
    var logging: Boolean = true,
    @field:NotNull
    var logFullUrl: Boolean = true,
    @field:NotNull
    var logMultipartBody: Boolean = true,
    @field:NotNull
    var tracing: Boolean = true,
    /**
     * 0 - бесконечные попытки
     * 1 - без повторных попыток
     * >1 - фиксированное количество повторных попыток
     */
    @field:PositiveOrZero
    var maxAttempts: Int = DEFAULT_MAX_ATTEMPTS,
    @field:NotNull
    @field:Valid
    @NestedConfigurationProperty
    var retryBackoff: RetryBackoffProperties = RetryBackoffProperties(),
    @field:NotNull
    var requestTiming: Boolean = false,
    @field:NotNull
    var readTimeout: Duration = DEFAULT_READ_TIMEOUT,
    @field:NotNull
    var connectTimeout: Duration = DEFAULT_CONNECT_TIMEOUT,
    @field:Valid
    @NestedConfigurationProperty
    var proxy: ProxyProperties? = null,
) {

    class RetryBackoffProperties(
        val delay: Duration = DEFAULT_RETRY_DELAY,
        val maxDelay: Duration? = null,
        /**
         * == 1 - фиксированный интервал между попытками
         * > 1 - экспоненциальный интервал между попытками
         */
        @field:Min(1)
        val multiplier: Double = DEFAULT_RETRY_MULTIPLIER,
        val random: Boolean = false,
    )

    class ProxyProperties(
        @field:NotNull
        var enabled: Boolean = true,
        @field:NotBlank
        var host: String,
        @field:Max(Short.MAX_VALUE.toLong())
        @field:Positive
        var port: Int,
        var username: String?,
        var password: String?,
        @field:NotNull
        var connectTimeout: Duration = DEFAULT_CONNECT_TIMEOUT,
    )

    enum class ClientType {
        INTERNAL,
        EXTERNAL,
    }
}
