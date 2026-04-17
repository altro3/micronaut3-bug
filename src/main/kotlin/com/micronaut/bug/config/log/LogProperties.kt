package com.micronaut.bug.config.log

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties("app.log")
class LogProperties(
    var mdcKeys: List<String> = emptyList(),
)
