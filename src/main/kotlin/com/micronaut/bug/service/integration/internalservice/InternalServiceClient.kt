package com.micronaut.bug.service.integration.internalservice

import com.micronaut.bug.client.DefaultHttpClient
import com.micronaut.bug.service.integration.internalservice.config.InternalServiceProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class InternalServiceClient(
    props: InternalServiceProperties,
    @Qualifier("internalServiceHttpClient")
    val httpClient: DefaultHttpClient,
) {

    private val endpoints = props.endpoints

    fun ping() {
        httpClient.sendRq(endpoints.ping)
            .toBodilessEntity()
    }
}
