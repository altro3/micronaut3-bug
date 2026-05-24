package com.altro.service2.config

import com.altro.common.util.api.config.DefWebConfig
import com.altro.common.util.api.json.JsonUtil
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Primary
import tools.jackson.databind.json.JsonMapper

@Configuration
class AppConfig {

    @Primary
    @Bean
    fun jsonMapper(): JsonMapper = JsonUtil.createMapper()

    @Configuration
    class WebConfig(jsonMapper: JsonMapper) : DefWebConfig(jsonMapper)
}
