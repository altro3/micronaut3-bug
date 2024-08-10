package com.micronaut.bug.controller

import io.micronaut.http.annotation.*
import io.micronaut.http.client.BlockingHttpClient
import io.micronaut.http.client.HttpClient
import io.micronaut.http.client.annotation.Client
import io.micronaut.scheduling.TaskExecutors
import io.micronaut.scheduling.annotation.ExecuteOn
import jakarta.annotation.Nullable

@ExecuteOn(TaskExecutors.BLOCKING)
@Controller
open class MyEntityController(
    @Client("local") httpClient: HttpClient
) {
    private val httpClient: BlockingHttpClient

    init {
        this.httpClient = httpClient.toBlocking()
    }

    @Post("/test{/apiVersion}")
    open fun test(
        @Header("X-Favor-Token") @Nullable xFavorToken: String? = null,
        @PathVariable("apiVersion") @Nullable apiVersion: MyEnum? = MyEnum.V5,
        @Header("Content-Type") @Nullable contentType: String? = "application/json"
    ): String? {
        return null
    }
}
