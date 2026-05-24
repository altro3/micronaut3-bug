package com.altro.service1.service.integration.service2.config

import com.altro.common.client.DefaultHttpClient
import com.altro.common.trace.NanoTracer
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import tools.jackson.databind.json.JsonMapper

@Configuration
@EnableConfigurationProperties(InternalServiceProperties::class)
class InternalServiceConfig {

    @Bean
    fun internalServiceHttpClient(
        props: InternalServiceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        jsonMapper: JsonMapper,
        nanoTracer: NanoTracer? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        jsonMapper = jsonMapper,
        tracer = nanoTracer,
    )
}
