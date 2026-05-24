package com.altro.common.log.config

import com.altro.common.log.LogReconfigurator
import com.altro.common.log.ServerLoggingFilter
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.env.Environment
import tools.jackson.databind.json.JsonMapper

@EnableConfigurationProperties(LogProperties::class)
@Configuration
class LogAutoConfig {

    @Bean
    fun logbackReconfigurator(
        environment: Environment,
        logProperties: LogProperties,
    ) = LogReconfigurator(
        environment = environment,
        props = logProperties,
    )

    @Bean
    fun unifiedLoggingFilterRegistration(
        jsonMapper: JsonMapper,
        props: LogProperties,
    ): FilterRegistrationBean<ServerLoggingFilter> {
        val registration = FilterRegistrationBean(
            ServerLoggingFilter(
                jsonMapper = jsonMapper,
                logProps = props,
            ),
        )
        registration.order = props.order
        return registration
    }
}
