package com.altro.service1.service.integration.extservice

import com.altro.common.client.DefaultHttpClient
import com.altro.service1.service.integration.extservice.config.ExtServiceProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class ExtServiceClient2(
    props: ExtServiceProperties,
    @Qualifier("extServiceHttpClient")
    val httpClient: DefaultHttpClient,
) {

    private val endpoints = props.endpoints

    fun ping() {
        httpClient.sendRq(endpoints.ping).toBodilessEntity()
    }
}
