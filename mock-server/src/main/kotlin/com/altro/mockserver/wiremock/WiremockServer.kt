package com.altro.mockserver.wiremock

import com.altro.mockserver.wiremock.config.WiremockProperties
import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.core.Options
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy

class WiremockServer(
    wiremockProps: WiremockProperties,
) {

    private val wireMockServer = WireMockServer(
        wireMockConfig()
            .port(wiremockProps.port)
            .useChunkedTransferEncoding(Options.ChunkedEncodingPolicy.NEVER)
    )

    @PostConstruct
    fun start() {
        wireMockServer.start()
        setupStubs()
        // Теперь WireMock будет писать в свой лог подробности несовпадения запросов (если будут)
    }

    @PreDestroy
    fun stop() {
        wireMockServer.stop()
    }

    private fun setupStubs() {
        InternalStubs.setupStubs(wireMockServer)
    }
}
