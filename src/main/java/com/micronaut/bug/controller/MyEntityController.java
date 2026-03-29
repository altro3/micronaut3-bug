package com.micronaut.bug.controller;

import com.micronaut.bug.config.SecurityContext;
import com.micronaut.bug.config.User;
import com.micronaut.bug.service.BusinessService;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@RequiredArgsConstructor
@Slf4j
@RestController
public class MyEntityController {

    private final BusinessService businessService;

    @PostMapping("/testMultipart/{pathVar}")
    public MyData testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestBody(required = false) MyData dataBody,
        @RequestPart(required = false) MyData data,
        @RequestPart(required = false) MultipartFile file
//        @CurrentUser User user
    ) throws IOException {
        log.info("body: {}", dataBody != null ? dataBody : "null");
        log.info("file: {}", file != null ? new String(file.getBytes(), StandardCharsets.UTF_8) : "null");
        log.info("pathVar: {}", pathVar);
        log.info("queryVar: {}", queryVar);

//        log.info("Текущий поток: {}", Thread.currentThread());
//        boolean isVirtual = Thread.currentThread().isVirtual();
//        log.info("Это виртуальный поток? {}", isVirtual);
//        var user2 = businessService.processOrder();

        if (dataBody != null) {
            dataBody.setUser(SecurityContext.getUser());
        }
        return dataBody;
    }

    @Data
    public static class MyData {

        private String name;
        private String secondName;
        private int age;
        public User user;
    }
}
