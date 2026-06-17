package com.altro.myorchestrator.config

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class McpConfig {

    @Bean
    fun chatClient(
        chatClientBuilder: ChatClient.Builder,
        mcpToolCallbackProvider: SyncMcpToolCallbackProvider,
    ): ChatClient =
        chatClientBuilder
            .defaultTools(mcpToolCallbackProvider)
            .build()
}
