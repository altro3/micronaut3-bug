package com.micronaut.bug.controller;

import com.micronaut.bug.config.User;
import com.micronaut.bug.config.UserContext;
import com.micronaut.bug.service.BusinessService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
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
    public MyData testMultipart(
        @PathVariable String pathVar,
        @RequestParam String queryVar,
        @RequestBody(required = false) MyData data
//        @RequestPart(required = false) MyData data,
//        @RequestPart(required = false) FilePart file
//        @CurrentUser User user
    ) {
        log.info("body: {}", data != null ? data : "null");
//        log.info("file: {}", file != null ? new String(readFilePart(file)) : "null");
        log.info("pathVar: {}", pathVar);
        log.info("queryVar: {}", queryVar);

        var user = UserContext.get();

        var user2 = businessService.processOrder(user);
        if (data != null) {
            data.user = user2;
        } else {
            data = new MyData();
            data.user = user2;
        }

        return data;
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

    private byte[] readFilePart(FilePart filePart) {
        return DataBufferUtils.join(filePart.content()) // Собираем все чанки в один Mono<DataBuffer>
            .map(dataBuffer -> {
                byte[] bytes = new byte[dataBuffer.readableByteCount()];
                dataBuffer.read(bytes);
                DataBufferUtils.release(dataBuffer); // ВАЖНО: освобождаем память Netty
                return bytes;
            })
            .block(); // Блокируем виртуальный поток до завершения чтения
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
