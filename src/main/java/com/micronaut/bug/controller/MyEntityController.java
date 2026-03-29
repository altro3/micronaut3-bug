package com.micronaut.bug.controller;

import com.micronaut.bug.config.User;
import com.micronaut.bug.service.BusinessService;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;

@RequiredArgsConstructor
@Slf4j
@RestController
public class MyEntityController {

    private final BusinessService businessService;

    @PostMapping("/testMultipart/{pathVar}")
    public Mono<MyData> testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestPart(required = false) MyData data,
        @RequestPart(required = false) FilePart file,
        @CurrentUser User user
    ) {
        return readFile(file) // Метод ниже возвращает Mono<String>
            .flatMap(fileString -> {
                log.info("body: {}", data != null ? data : "null");
                log.info("file: {}", fileString != null ? fileString : "null");
                log.info("pathVar: {}", pathVar);
                log.info("queryVar: {}", queryVar);

                var user2 = businessService.processOrder(user);
                if (data != null) {
                    data.user = user2;
                }

                return Mono.justOrEmpty(data);
            });
    }

    private Mono<String> readFile(FilePart file) {
        return DataBufferUtils.join(file.content())
            .map(db -> {
                var bytes = new byte[db.readableByteCount()];
                db.read(bytes);
                DataBufferUtils.release(db);
                return new String(bytes, StandardCharsets.UTF_8);
            });
    }

    @Data
    public static class MyData {

        private String name;
        private String secondName;
        private int age;
        public User user;
    }
}
