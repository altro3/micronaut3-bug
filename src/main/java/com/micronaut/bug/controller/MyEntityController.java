package com.micronaut.bug.controller;

import com.micronaut.bug.api.EnumParam;
import com.micronaut.bug.service.MyEntityService;
import io.micronaut.core.annotation.Nullable;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.Head;
import io.micronaut.http.annotation.Header;
import io.micronaut.http.annotation.Part;
import io.micronaut.http.annotation.PathVariable;
import io.micronaut.http.annotation.Post;
import io.micronaut.http.annotation.QueryValue;
import io.micronaut.http.multipart.CompletedFileUpload;
import io.micronaut.scheduling.TaskExecutors;
import io.micronaut.scheduling.annotation.ExecuteOn;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@ExecuteOn(TaskExecutors.BLOCKING)
@Slf4j
@RequiredArgsConstructor
@Controller("/api")
public class MyEntityController {

    private final MyEntityService entityService;

    @Head("/")
    void fileAvailable(
        @Header("X-TEST-crc32") @NotNull String xtESTCrc32,
        @Header("X-TEST-sha512") @NotNull String xtESTSha512,
        @Header("X-TEST-size") @NotNull Long xtESTSize
    ) {
        log.info("xtESTCrc32: {}", xtESTCrc32);
    }

    @Get("/test/{param}")
    public void test(@PathVariable Integer param) {
        log.info("test");
    }

    @Get("/test2/{param}")
    public EnumParam test2(EnumParam param, @QueryValue @Nullable EnumParam qParam) {
        return param;
    }

    @Post(value = "/testMultipart", consumes = MediaType.MULTIPART_FORM_DATA)
    public void testMultipart(String fileName, CompletedFileUpload file) throws IOException {
        log.info("fileName: {}", fileName.getBytes());
    }

    @Post(value = "/file", consumes = MediaType.MULTIPART_FORM_DATA)
    public void endpoint2(@Part Map<String, CompletedFileUpload> files) {
        log.info("endpoint2");
    }

    @Post(value = "/file2", consumes = MediaType.MULTIPART_FORM_DATA)
    public void endpoint3(List<CompletedFileUpload> files) {
        log.info("endpoint3");
    }

}
