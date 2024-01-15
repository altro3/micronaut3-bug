package com.micronaut.bug.controller

import com.micronaut.bug.api.Animal
import com.micronaut.bug.api.Bird
import com.micronaut.bug.api.ColorEnum
import com.micronaut.bug.config.TestProperties
import io.micronaut.http.HttpRequest
import io.micronaut.http.annotation.Controller
import io.micronaut.http.annotation.Get
import io.micronaut.http.client.BlockingHttpClient
import io.micronaut.http.client.HttpClient
import io.micronaut.http.client.annotation.Client
import io.micronaut.scheduling.TaskExecutors
import io.micronaut.scheduling.annotation.ExecuteOn
import java.math.BigDecimal

@ExecuteOn(TaskExecutors.BLOCKING)
@Controller
class MyEntityController(
        @Client("local") httpClient: HttpClient,
        private val testProperties: TestProperties,
) {
    private val httpClient: BlockingHttpClient

    init {
        this.httpClient = httpClient.toBlocking()
    }

    @Get("/test")
    fun test(): String {
        println("Test properties enabled = ${testProperties.enabled}")
        return "OK"
    }

    @Get("/start")
    fun start() {
        val bird = Bird(
                numWings = 2,
                beakLength = BigDecimal.valueOf(12, 1),
                featherDescription = "Large blue and white feathers",
        )
        bird.color = ColorEnum.BLUE
        val resposne = httpClient.retrieve(HttpRequest.POST<Animal>("/test", bird), String::class.java)
        println("Response: $resposne")
    }
}
