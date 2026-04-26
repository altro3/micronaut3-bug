package com.micronaut.bug.config.log

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.config.ServerLoggingFilter
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.env.Environment

@EnableConfigurationProperties(LogProperties::class)
@Configuration
class LogAutoConfig {

    @Bean
    fun logbackReconfigurator(environment: Environment, logProperties: LogProperties) =
        LogReconfigurator(environment, logProperties)

    @Bean
    fun unifiedLoggingFilterRegistration(
        objectMapper: ObjectMapper,
        props: LogProperties,
        @Value("\${spring.application.name}")
        appName: String,
    ): FilterRegistrationBean<ServerLoggingFilter> {
        val registration = FilterRegistrationBean<ServerLoggingFilter>()
        registration.filter = ServerLoggingFilter(objectMapper, props, appName)
        registration.order = props.order
        return registration
    }
}
