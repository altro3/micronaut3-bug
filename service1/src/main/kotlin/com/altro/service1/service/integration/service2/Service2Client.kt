package com.altro.service1.service.integration.service2

import com.altro.common.client.DefaultHttpClient
import com.altro.service1.service.integration.service2.config.InternalServiceProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class Service2Client(
    props: InternalServiceProperties,
    @Qualifier("internalServiceHttpClient")
    val httpClient: DefaultHttpClient,
) {

    private val endpoints = props.endpoints

    fun ping() {
        httpClient.sendRq(endpoints.ping)
            .toBodilessEntity()
    }

    fun error() {
        httpClient.sendRq(endpoints.error)
            .toBodilessEntity()
    }
}
