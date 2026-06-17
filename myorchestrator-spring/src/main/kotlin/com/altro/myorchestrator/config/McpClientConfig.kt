package com.altro.myorchestrator.config

import io.modelcontextprotocol.client.McpAsyncClient
import io.modelcontextprotocol.client.McpClient
import io.modelcontextprotocol.spec.McpSchema.Implementation
import org.springframework.ai.mcp.client.common.autoconfigure.NamedClientMcpTransport
import org.springframework.ai.mcp.client.common.autoconfigure.properties.McpClientCommonProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class McpClientConfig {

    @Bean
    fun adBrokerMcpClient(
        mcpProps: McpClientCommonProperties,
        @Qualifier("streamableHttpHttpClientTransports")
        namedTransports: List<NamedClientMcpTransport>,
    ): McpAsyncClient {
        val transport = namedTransports.find { it.name() == "adbroker-automation" }
            ?: error("McpTransport 'adbroker-automation' не найден в контексте")

        // 🎯 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Переключаем фабрику с .sync() на .async()
        val client = McpClient.async(transport.transport())
            .requestTimeout(mcpProps.requestTimeout)
            .clientInfo(
                Implementation(
                    mcpProps.name,
                    mcpProps.name,
                    mcpProps.version,
                    null,
                    null,
                    null,
                )
            )
            .build()

        if (mcpProps.isInitialized) {
            client.initialize().subscribe()
        }

        return client
    }

    /*
        @Bean
        fun adbrokerMcpClient(
            mcpProps: McpClientCommonProperties,
            @Qualifier("streamableHttpHttpClientTransports")
            namedTransports: List<NamedClientMcpTransport>,
        ): McpSyncClient {
            val transport = namedTransports.find { it.name() == "adbroker-automation" }
                ?: error("McpTransport 'adbroker-automation' не найден в контексте")

            val client = McpClient.sync(transport.transport())
                .requestTimeout(mcpProps.requestTimeout)
                .clientInfo(
                    Implementation(
                        mcpProps.name,
                        mcpProps.name,
                        mcpProps.version,
                        null,
                        null,
                        null,
                    )
                )
                .build()

            if (mcpProps.isInitialized) {
                client.initialize()
            }

            return client
        }
    */
}
