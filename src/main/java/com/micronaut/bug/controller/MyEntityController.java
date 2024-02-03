package com.micronaut.bug.controller;

import java.math.BigDecimal;

import io.micronaut.http.HttpRequest;
import io.micronaut.http.annotation.Body;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.Post;
import io.micronaut.http.client.BlockingHttpClient;
import io.micronaut.http.client.HttpClient;
import io.micronaut.http.client.annotation.Client;

import lombok.extern.slf4j.Slf4j;

import com.micronaut.bug.api.Animal;
import com.micronaut.bug.api.Bird;
import com.micronaut.bug.api.ColorEnum;
import com.micronaut.bug.api.SimpleDto;
import com.micronaut.bug.service.MyEntityService;

@Slf4j
@Controller
public class MyEntityController {

    private final MyEntityService entityService;
    private final BlockingHttpClient httpClient;

    public MyEntityController(MyEntityService entityService, @Client("local") HttpClient httpClient) {
        this.entityService = entityService;
        this.httpClient = httpClient.toBlocking();
    }

    @Post("/test")
    public SimpleDto test(@Body SimpleDto animal) {
        return animal;
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
