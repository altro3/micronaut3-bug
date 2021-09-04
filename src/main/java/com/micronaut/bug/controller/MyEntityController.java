package com.micronaut.bug.controller;

import com.micronaut.bug.service.MyEntityService;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Controller
public class MyEntityController {

    private final MyEntityService entityService;

    @Get(value = "/test", produces = MediaType.TEXT_HTML)
    public String test() {
        entityService.loadFromDb();
        return "OK!";
    }
}
