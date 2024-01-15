package com.micronaut.bug.config

import io.micronaut.context.annotation.ConfigurationInject
import io.micronaut.context.annotation.ConfigurationProperties

@ConfigurationProperties("testing")
data class TestProperties @ConfigurationInject constructor(
    val enabled: Boolean = true
)
