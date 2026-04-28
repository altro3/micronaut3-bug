package com.micronaut.bug.config.trace.config

import com.micronaut.bug.config.trace.TempoExporter
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.net.http.HttpClient

@ConditionalOnBooleanProperty("app.trace.tempo.enabled", matchIfMissing = true)
@Configuration
@EnableConfigurationProperties(TraceProperties::class)
class TraceAutoConfig {

    @Bean
    fun tempoHttpClient(properties: TraceProperties) =
        HttpClient.newBuilder()
            .connectTimeout(properties.connectTimeout)
            .build()


    @Bean
    fun tempoExporter(
        properties: TraceProperties,
        @Value("\${spring.application.name}")
        appName: String,
        @Qualifier("tempoHttpClient")
        httpClient: HttpClient,
    ) = TempoExporter(
        appName = appName,
        properties = properties,
        httpClient = httpClient,
    )
}
