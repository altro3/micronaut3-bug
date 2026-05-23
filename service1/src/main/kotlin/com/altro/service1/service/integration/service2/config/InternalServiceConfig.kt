package com.altro.service1.service.integration.service2.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.altro.common.client.DefaultHttpClient
import com.altro.common.trace.NanoTracer
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
@EnableConfigurationProperties(InternalServiceProperties::class)
class InternalServiceConfig {

    @Bean
    fun internalServiceHttpClient(
        props: InternalServiceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        objectMapper: ObjectMapper,
        nanoTracer: NanoTracer? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        objectMapper = objectMapper,
        tracer = nanoTracer,
    )
}
