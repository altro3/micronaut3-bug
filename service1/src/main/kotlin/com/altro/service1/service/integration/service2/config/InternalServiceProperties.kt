package com.altro.service1.service.integration.service2.config

import com.altro.common.client.Endpoint
import com.altro.common.client.HttpClientProperties
import jakarta.validation.Valid
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty

@ConfigurationProperties("app.integration.service2")
class InternalServiceProperties(
    @NestedConfigurationProperty
    val http: HttpClientProperties = HttpClientProperties(),

    @NestedConfigurationProperty
    val endpoints: Endpoints,
) {

    data class Endpoints(
        @field:Valid
        val ping: Endpoint,
        @field:Valid
        val error: Endpoint,
    )
}