package com.altro.mcp.service.integration.serviceyad.config

import com.altro.common.client.DefaultHttpClient
import com.altro.common.trace.NanoTracer
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
@EnableConfigurationProperties(ServiceYadProperties::class)
class ServiceYadConfig {

    @Bean
    fun serviceYadHttpClient(
        props: ServiceYadProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        tracer: NanoTracer? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        tracer = tracer,
    )
}
