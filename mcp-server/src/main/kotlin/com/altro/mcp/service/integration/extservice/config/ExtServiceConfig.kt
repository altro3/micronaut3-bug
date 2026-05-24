package com.altro.service2.service.integration.extservice.config

import com.altro.common.client.DefaultHttpClient
import com.altro.common.trace.NanoTracer
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import tools.jackson.databind.json.JsonMapper

@Configuration
@EnableConfigurationProperties(ExtServiceProperties::class)
class ExtServiceConfig {

    @Bean
    fun extServiceHttpClient(
        props: ExtServiceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        tracer: NanoTracer? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        tracer = tracer,
    )
}
