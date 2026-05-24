package com.altro.mockserver.wiremock.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties("app.wiremock")
class WiremockProperties(
    var enabled: Boolean = true,
    var port: Int,
)