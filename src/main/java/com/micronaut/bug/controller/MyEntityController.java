package com.micronaut.bug.controller;

import com.micronaut.bug.service.MyEntityService;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Get;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
@RequiredArgsConstructor
//@Controller
public class MyEntityController {

    private final MyEntityService entityService;

    @Get(value = "/load", produces = MediaType.TEXT_HTML)
    public String test() {
        entityService.loadFromDb();
        return "OK!";
    }

    @Get(value = "/save", produces = MediaType.TEXT_HTML)
    public String save() {
        entityService.save();
        return "OK!";
    }

    @Get(value = "/props", produces = MediaType.TEXT_HTML)
    public List<String> getPropertiesBykey() {
        return entityService.getPropertiesByKey("key1");
    }
}
