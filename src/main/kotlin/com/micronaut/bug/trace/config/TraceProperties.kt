package com.micronaut.bug.trace.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty
import java.net.URI
import java.time.Duration

@ConfigurationProperties("app.trace")
class TraceProperties(
    val enabled: Boolean = true,
    val slowRequestThreshold: Duration = Duration.ofSeconds(10),
    val sampleRate: Double = 1.0,
    @NestedConfigurationProperty
    val exporter: ExporterProperties = ExporterProperties(),
    val propagationHeaders: Set<String> = setOf(),
) {

    class ExporterProperties(
        val enabled: Boolean = true,
        val url: URI = URI.create("http://localhost:4318/v1/traces"),
        val connectTimeout: Duration = Duration.ofSeconds(2),
        val requestTimeout: Duration = Duration.ofSeconds(2),
        val batchSize: Int = 512,
        val flushInterval: Duration = Duration.ofSeconds(2),
        val queueCapacity: Int = 10000,
        val shutdownTimeout: Duration = Duration.ofSeconds(5),
        val retryInterval: Duration = Duration.ofSeconds(1)
    )
}
