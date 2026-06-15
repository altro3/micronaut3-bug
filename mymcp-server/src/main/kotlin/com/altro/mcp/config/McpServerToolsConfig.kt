package com.altro.mcp.config

import com.altro.mcp.service.AdvertisingMcpTools
import org.springframework.ai.tool.ToolCallback
import org.springframework.ai.tool.method.MethodToolCallbackProvider
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class McpServerToolsConfig {

    @Bean
    fun adBrokerMethodToolCallbacks(advertisingMcpTools: AdvertisingMcpTools): List<ToolCallback> =
        MethodToolCallbackProvider.builder()
            .toolObjects(advertisingMcpTools)
            .build()
            .toolCallbacks
            .toList()
}
