package com.altro.service2.service.integration.extservice

import com.altro.common.client.DefaultHttpClient
import com.altro.service2.service.integration.extservice.api.MyDataRequest
import com.altro.service2.service.integration.extservice.api.MyDataResponse
import com.altro.service2.service.integration.extservice.config.ExtServiceProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class ExtServiceClient(
    props: ExtServiceProperties,
    @Qualifier("extServiceHttpClient")
    val httpClient: DefaultHttpClient,
) {

    private val endpoints = props.endpoints

    fun updateData(request: MyDataRequest): MyDataResponse? =
        httpClient.sendRq(
            endpoint = endpoints.updateData,
            responseClass = MyDataResponse::class.java,
            rqBody = request
        )
}
