package com.micronaut.bug.controller;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@Slf4j
@RestController
public class MyEntityController {

    @PostMapping("/testMultipart/{pathVar}")
    public MyData testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestPart(required = false) MyData data,
        @RequestPart(required = false) FilePart file
    ) throws IOException {
        log.info("body: {}", data != null ? data : "null");
        log.info("file: {}", file != null ? new String(file.content().blockLast().asInputStream().readAllBytes()) : "null");
        log.info("pathVar: {}", pathVar);
        log.info("queryVar: {}", queryVar);
        return data;
    }

    @Data
    public static class MyData {

        private String name;
        private String secondName;
        private int age;
    }
}
