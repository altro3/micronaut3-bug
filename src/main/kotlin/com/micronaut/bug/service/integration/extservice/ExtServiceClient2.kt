package com.micronaut.bug.service.integration.extservice

import com.micronaut.bug.client.DefaultHttpClient
import com.micronaut.bug.service.integration.extservice.api.MyDataRequest
import com.micronaut.bug.service.integration.extservice.api.MyDataResponse
import com.micronaut.bug.service.integration.extservice.config.ExtServiceProperties
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

    fun updateData(request: MyDataRequest): MyDataResponse? {
        return httpClient.sendRq(
            endpoint = endpoints.updateData,
            responseClass = MyDataResponse::class.java,
            rqBody = request
        )
    }
}
