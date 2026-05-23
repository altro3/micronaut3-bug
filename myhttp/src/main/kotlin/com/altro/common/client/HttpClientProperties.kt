package com.altro.common.client

import com.altro.common.client.HttpClientProperties.ClientType.INTERNAL
import com.altro.common.client.HttpClientUtils.DEFAULT_CONNECT_TIMEOUT
import com.altro.common.client.HttpClientUtils.DEFAULT_MAX_ATTEMPTS
import com.altro.common.client.HttpClientUtils.DEFAULT_READ_TIMEOUT
import com.altro.common.client.HttpClientUtils.DEFAULT_RETRY_DELAY
import com.altro.common.client.HttpClientUtils.DEFAULT_RETRY_MULTIPLIER
import jakarta.validation.Valid
import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import jakarta.validation.constraints.PositiveOrZero
import org.springframework.boot.context.properties.NestedConfigurationProperty
import org.springframework.util.unit.DataSize
import org.springframework.validation.annotation.Validated
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
    var compress: Boolean = false,
    @field:NotNull
    var tracing: Boolean = true,
    @field:NotNull
    @field:Valid
    @NestedConfigurationProperty
    var log: LogProperties = LogProperties(),
    @field:PositiveOrZero
    var maxAttempts: Int = DEFAULT_MAX_ATTEMPTS,
    @field:NotNull
    @field:Valid
    @NestedConfigurationProperty
    var retryBackoff: RetryBackoffProperties = RetryBackoffProperties(),
    @field:NotNull
    var readTimeout: Duration = DEFAULT_READ_TIMEOUT,
    @field:NotNull
    var connectTimeout: Duration = DEFAULT_CONNECT_TIMEOUT,
    @field:Valid
    @NestedConfigurationProperty
    var proxy: ProxyProperties? = null,
) {

    class LogProperties(
        @field:NotNull
        var enabled: Boolean = true,
        @field:NotNull
        var fullUrl: Boolean = true,
        @field:NotNull
        var maxPayloadSize: DataSize = DataSize.ofMegabytes(15),
        @field:Min(-1)
        var limitLogSize: Int = 8192,
        @field:Positive
        var truncateChunkSize: Int = 4096,
        @field:NotNull
        var prettyPrint: Boolean = true,
        var slowThreshold: Duration = Duration.ofSeconds(10),
    )

    class RetryBackoffProperties(
        val delay: Duration = DEFAULT_RETRY_DELAY,
        val maxDelay: Duration? = null,
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
