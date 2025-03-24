package com.micronaut.bug.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
public class MyEntityController {

    @PostMapping(value = "/testMultipart", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void testMultipart(String fileName, MultipartFile file) throws IOException {
        log.info("fileName: {}", fileName.getBytes());
    }
}
