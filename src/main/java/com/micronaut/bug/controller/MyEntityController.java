package com.micronaut.bug.controller;

import java.math.BigDecimal;

import com.example.openapi.api.DefaultApi;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.annotation.Body;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.Head;
import io.micronaut.http.annotation.Header;
import io.micronaut.http.annotation.Post;
import io.micronaut.http.client.BlockingHttpClient;
import io.micronaut.http.client.HttpClient;
import io.micronaut.http.client.annotation.Client;
import io.micronaut.scheduling.TaskExecutors;
import io.micronaut.scheduling.annotation.ExecuteOn;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

import com.micronaut.bug.api.Animal;
import com.micronaut.bug.api.Bird;
import com.micronaut.bug.api.ColorEnum;
import com.micronaut.bug.service.MyEntityService;

@ExecuteOn(TaskExecutors.BLOCKING)
@Slf4j
@Controller
public class MyEntityController {

    private final MyEntityService entityService;
    private final BlockingHttpClient httpClient;
    private final DefaultApi defaultApi;

    public MyEntityController(MyEntityService entityService, @Client("local") HttpClient httpClient, DefaultApi defaultApi) {
        this.entityService = entityService;
        this.httpClient = httpClient.toBlocking();
        this.defaultApi = defaultApi;
    }

    @Head("/")
    void fileAvailable(
            @Header("X-TEST-crc32") @NotNull String xtESTCrc32,
            @Header("X-TEST-sha512") @NotNull String xtESTSha512,
            @Header("X-TEST-size") @NotNull Long xtESTSize
    ) {
        log.info("xtESTCrc32: {}", xtESTCrc32);
    }

    @Get("/test")
    public void test() {
        defaultApi.fileAvailable("test1", "test2", 123L);
    }

    @Get("/start")
    public void start() {

        var bird = new Bird()
                .numWings(2)
                .beakLength(BigDecimal.valueOf(12, 1))
                .featherDescription("Large blue and white feathers")
                .color(ColorEnum.BLUE);

        var resposne = httpClient.retrieve(HttpRequest.<Animal>POST("/test", bird), Animal.class);
        log.info("Response: {}", resposne);
    }
}
