package com.micronaut.bug.service.integration.extservice.config

import com.micronaut.bug.client.Endpoint
import com.micronaut.bug.client.HttpClientProperties
import jakarta.validation.Valid
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty

@ConfigurationProperties("app.integration.ext-service")
class ExtServiceProperties(
    @NestedConfigurationProperty
    val http: HttpClientProperties = HttpClientProperties(),

    @NestedConfigurationProperty
    val endpoints: Endpoints,
) {

    data class Endpoints(
        @field:Valid
        val ping: Endpoint,
        @field:Valid
        val processFile: Endpoint,
        @field:Valid
        val updateData: Endpoint,
        @field:Valid
        val complexMultipart: Endpoint,
    )
}