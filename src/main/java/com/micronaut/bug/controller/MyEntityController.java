package com.micronaut.bug.controller;

import com.micronaut.bug.config.SecurityContext;
import com.micronaut.bug.config.User;
import com.micronaut.bug.service.BusinessService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RequiredArgsConstructor
@Slf4j
@RestController
public class MyEntityController {

    private final BusinessService businessService;

    @PostMapping("/testMultipart/{pathVar}")
    public MyData testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestBody(required = false) MyData data
//        @RequestPart(required = false) MyData data,
//        @RequestPart(required = false) MultipartFile file
//        @CurrentUser User user
    ) {
        log.info("body: {}", data != null ? data : "null");
//        log.info("file: {}", file != null ? new String(file.getBytes(), StandardCharsets.UTF_8) : "null");
        log.info("pathVar: {}", pathVar);
        log.info("queryVar: {}", queryVar);

        var user = SecurityContext.getUser();

        var user2 = businessService.processOrder(user);
        if (data != null) {
            data.user = user2;
        } else {
            data = new MyData();
            data.user = user2;
        }

        return data;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MyData {

        private String name;
        private String secondName;
        private int age;
        public User user;
    }
}
