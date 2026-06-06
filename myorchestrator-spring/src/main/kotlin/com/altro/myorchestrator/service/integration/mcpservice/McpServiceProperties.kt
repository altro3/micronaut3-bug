package com.altro.myorchestrator.service.integration.mcpservice

import com.altro.common.client.HttpClientProperties
import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties("app.integration.mcp-server")
data class McpServiceProperties(
    val initTimeout: Duration = Duration.ofSeconds(15),
    val http: HttpClientProperties,
    val endpoints: Endpoints
) {
    data class Endpoints(
        val sse: Endpoint
    )

    data class Endpoint(
        val url: String
    )
}