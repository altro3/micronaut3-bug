package com.micronaut.bug.config.log

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.core.Ordered

@ConfigurationProperties("app.log")
class LogProperties(
    var enabledControllerLogging: Boolean = true,
    var order: Int = Ordered.HIGHEST_PRECEDENCE,
    var skipActuator: Boolean = true,
    var mdcKeys: List<String> = emptyList(),
)
