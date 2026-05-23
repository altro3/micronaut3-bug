package com.altro.common.log.config

import com.fasterxml.jackson.databind.ObjectMapper
import com.altro.common.log.LogReconfigurator
import com.altro.common.log.ServerLoggingFilter
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.env.Environment

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
        objectMapper: ObjectMapper,
        props: LogProperties,
    ): FilterRegistrationBean<ServerLoggingFilter> {
        val registration = FilterRegistrationBean<ServerLoggingFilter>()
        registration.filter = ServerLoggingFilter(
            objectMapper = objectMapper,
            logProps = props,
        )
        registration.order = props.order
        return registration
    }
}
