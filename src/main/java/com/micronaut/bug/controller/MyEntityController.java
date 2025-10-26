package com.micronaut.bug.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
public class MyEntityController {

    @PostMapping("/testMultipart/{pathVar}")
    public String testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestBody String body
    ) {
        log.info("body: {}", body);
        log.info("pathVar: {}", pathVar);
        log.info("queryVar: {}", queryVar);
        return body;
    }
}
