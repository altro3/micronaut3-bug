package com.micronaut.bug.service.integration.internalservice.config

import com.micronaut.bug.client.Endpoint
import com.micronaut.bug.client.HttpClientProperties
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
    )
}