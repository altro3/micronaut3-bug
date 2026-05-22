package com.micronaut.bug.trace.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty
import org.springframework.util.unit.DataSize
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
    var stackTraceMaxLines: Int = 10,
    var stackTraceRootCauseFull: Boolean = true,
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
        val retryInterval: Duration = Duration.ofSeconds(1),
        var useGzip: Boolean = true,
        var compressionThreshold: DataSize = DataSize.ofKilobytes(4),
        val maxSenders: Int = 5,
    )
}
