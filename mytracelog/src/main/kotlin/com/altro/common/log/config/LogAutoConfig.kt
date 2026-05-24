package com.altro.common.log.config

import com.altro.common.log.LogFormatter
import com.altro.common.log.LogReconfigurator
import com.altro.common.log.RequestWrapperFactory
import com.altro.common.log.ServerLoggingFilter
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.core.env.Environment
import tools.jackson.databind.json.JsonMapper

@ConditionalOnBooleanProperty("app.log.enabled", matchIfMissing = true)
@EnableConfigurationProperties(LogProperties::class)
@AutoConfiguration
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
        logProps: LogProperties,
    ): FilterRegistrationBean<ServerLoggingFilter> {
        val registration = FilterRegistrationBean(
            ServerLoggingFilter(
                formatter = LogFormatter(jsonMapper),
                logProps = logProps,
            ),
        )
        registration.order = logProps.order
        return registration
    }
}
