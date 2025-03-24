package com.micronaut.bug.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
public class MyEntityController {

    @PostMapping(value = "/testMultipart/{pathVar}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void testMultipart(WrappedRq rq) {
        log.info("fileName: {}", rq.fileName.getBytes());
    }

    public record WrappedRq(
        @PathVariable
        String pathVar,
        @RequestParam
        String queryVar,
        // Part from multipart body
        String fileName,
        // Part from multipart body
        MultipartFile file
    ) {}
}
