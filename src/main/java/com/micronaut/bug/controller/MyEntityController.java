package com.micronaut.bug.controller;

import com.micronaut.bug.service.MyService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RequiredArgsConstructor
@Slf4j
@RestController
public class MyEntityController {

    private final MyService myService;


    @PostMapping("/video")
    public void uploadVideo(@Valid UploadVideoRq uploadVideoRq) throws Exception {
        myService.uploadVideo(uploadVideoRq);
    }
    @PostMapping(value = "/testMultipart/{pathVar}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void testMultipart(@Valid WrappedRq rq) {
        log.info("fileName: {}", rq.fileName.getBytes());
    }

    public record UploadVideoRq(
        Long creativeId,
        MultipartFile file,
        int height,
        int width,
        int x,
        int y
    ) {
    }

    public record WrappedRq(
        String pathVar,
        String queryVar,
        // Part from multipart body
        String fileName,
        // Part from multipart body
        @NotNull
        MultipartFile file
    ) {
    }
}
