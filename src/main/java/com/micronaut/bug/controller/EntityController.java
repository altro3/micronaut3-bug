package com.micronaut.bug.controller;

import com.micronaut.bug.service.entity.service.EntityService;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Controller
public class EntityController {

    private final EntityService entityService;

    @Get(value = "/test", produces = MediaType.TEXT_HTML)
    public String test() {
        entityService.loadFromDb();
        return "OK!";
    }
}
