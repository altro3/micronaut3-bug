package com.altro.myorchestrator.service.integration.mcpservice

import io.modelcontextprotocol.client.McpClient
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport
import io.modelcontextprotocol.json.jackson3.JacksonMcpJsonMapper
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import tools.jackson.databind.json.JsonMapper
import java.net.InetSocketAddress
import java.net.ProxySelector
import java.net.http.HttpClient

//@Configuration
//@EnableConfigurationProperties(McpServiceProperties::class)
class McpServiceConfig {

    @Bean
    fun mcpSyncClient(
        properties: McpServiceProperties,
        jsonMapper: JsonMapper,
    ): McpSyncClient {
        val clientBuilder = HttpClient.newBuilder()
            .connectTimeout(properties.http.connectTimeout)

        val proxyProps = properties.http.proxy
        if (proxyProps != null && proxyProps.enabled) {
            clientBuilder.proxy(
                ProxySelector.of(
                    InetSocketAddress(proxyProps.host, proxyProps.port)
                )
            )
        }

        val transport = HttpClientStreamableHttpTransport.builder(properties.endpoints.sse.url)
            .clientBuilder(clientBuilder)
            .jsonMapper(JacksonMcpJsonMapper(jsonMapper))
            .httpRequestCustomizer { builder, _, _, _, _ ->
                builder.timeout(properties.http.readTimeout)
            }
            .build()

        val client = McpClient.sync(transport)
            .requestTimeout(properties.http.readTimeout)
            .initializationTimeout(properties.initTimeout)
            .enableCallToolSchemaCaching(true)
            .build()

        client.initialize()
        return client
    }
}


