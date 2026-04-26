package com.micronaut.bug.config.trace

import org.springframework.boot.context.properties.ConfigurationProperties
import java.net.URI
import java.time.Duration

@ConfigurationProperties(prefix = "app.trace.tempo")
data class TraceProperties(
    /**
     * Включить/выключить экспорт в Tempo.
     */
    val enabled: Boolean = false,
    /**
     * URL эндпоинта Tempo (OTLP HTTP Protobuf).
     */
    val url: URI = URI.create("http://localhost:4318/v1/traces"),
    /**
     * Таймаут на установку соединения.
     */
    val connectTimeout: Duration = Duration.ofSeconds(2),
)
