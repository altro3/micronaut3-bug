package com.altro.mockserver.wiremock.config

import com.altro.mockserver.wiremock.WiremockServer
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@ConditionalOnBooleanProperty("app.wiremock.enabled", matchIfMissing = true)
@Configuration
@EnableConfigurationProperties(WiremockProperties::class)
class WiremockConfig {

    @Bean
    fun wiremockServer(props: WiremockProperties) =
        WiremockServer(props)
}