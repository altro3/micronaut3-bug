package com.micronaut.bug.controller;

import com.micronaut.bug.api.Animal;
import com.micronaut.bug.api.Bird;
import com.micronaut.bug.api.ColorEnum;
import com.micronaut.bug.service.MyEntityService;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.Head;
import io.micronaut.http.annotation.Header;
import io.micronaut.http.client.BlockingHttpClient;
import io.micronaut.http.client.HttpClient;
import io.micronaut.http.client.annotation.Client;
import io.micronaut.scheduling.TaskExecutors;
import io.micronaut.scheduling.annotation.ExecuteOn;
import jakarta.validation.constraints.NotNull;
import lombok.extern.slf4j.Slf4j;

import java.math.BigDecimal;
import java.util.Optional;
import java.util.OptionalInt;

@ExecuteOn(TaskExecutors.BLOCKING)
@Slf4j
@Controller
public class MyEntityController {

    private final MyEntityService entityService;
    private final BlockingHttpClient httpClient;

    public MyEntityController(MyEntityService entityService, @Client("local") HttpClient httpClient) {
        this.entityService = entityService;
        this.httpClient = httpClient.toBlocking();
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
    public Optional<Integer> test() {
        return Optional.of(100);
    }

    @Get("/test2")
    public OptionalInt test2() {
        return OptionalInt.of(100);
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
