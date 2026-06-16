package com.altro.mcp.service.integration.serviceyad.config

import com.altro.common.client.Endpoint
import com.altro.common.client.HttpClientProperties
import jakarta.validation.Valid
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty

@ConfigurationProperties("app.integration.service-yad")
class ServiceYadProperties(
    @NestedConfigurationProperty
    val http: HttpClientProperties = HttpClientProperties(),
    @NestedConfigurationProperty
    val endpoints: Endpoints,
) {
    data class Endpoints(
        @field:Valid
        val createDraft: Endpoint,
        @field:Valid
        val bindRegions: Endpoint,
        @field:Valid
        val submitCreative: Endpoint,
        @field:Valid
        val getRegions: Endpoint,
    )
}
