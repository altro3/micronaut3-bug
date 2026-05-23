package com.altro.service2.service.integration.extservice.config

import com.altro.common.client.Endpoint
import com.altro.common.client.HttpClientProperties
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
        val updateData: Endpoint,
    )
}