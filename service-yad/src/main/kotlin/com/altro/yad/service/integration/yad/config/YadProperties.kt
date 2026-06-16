package com.altro.yad.service.integration.yad.config

import com.altro.common.client.Endpoint
import com.altro.common.client.HttpClientProperties
import jakarta.validation.Valid
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty

@ConfigurationProperties("app.integration.yad")
class YadProperties(
    @NestedConfigurationProperty
    val http: HttpClientProperties = HttpClientProperties(),
    @NestedConfigurationProperty
    val endpoints: Endpoints,
) {
    data class Endpoints(
        @field:Valid
        val getRegions: Endpoint,
        @field:Valid
        val createCampaign: Endpoint,
    )
}
