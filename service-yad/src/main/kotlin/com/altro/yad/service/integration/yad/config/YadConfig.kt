package com.altro.yad.service.integration.yad.config

import com.altro.common.client.DefaultHttpClient
import com.altro.common.trace.NanoTracer
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
@EnableConfigurationProperties(YadProperties::class)
class YadConfig {

    @Bean
    fun yadHttpClient(
        props: YadProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        tracer: NanoTracer? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        tracer = tracer,
    )
}
