package com.micronaut.bug.service.integration.extservice.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.client.DefaultHttpClient
import com.micronaut.bug.config.trace.TempoExporter
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
@EnableConfigurationProperties(ExtServiceProperties::class)
class ExtServiceConfig {

    @Bean
    fun extServiceHttpClient(
        props: ExtServiceProperties,
        @Value("\${spring.application.name}")
        appName: String,
        objectMapper: ObjectMapper,
        tempoExporter: TempoExporter? = null,
    ) = DefaultHttpClient(
        senderAppName = appName,
        httpClientProperties = props.http,
        objectMapper = objectMapper,
        tempoExporter = tempoExporter,
    )
}