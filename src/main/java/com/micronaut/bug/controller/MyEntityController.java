package com.micronaut.bug.controller;

import com.fasterxml.jackson.annotation.JsonProperty;
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
import io.micronaut.serde.annotation.Serdeable;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

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

    @Get("/record")
    public CostCenter endpoint3() {
        return new CostCenter(
            UUID.randomUUID(),
            "this is name",
            "this is code",
            "foreignKey",
            UUID.randomUUID(),
            LocalDateTime.now(),
            UUID.randomUUID(),
            LocalDateTime.now(),
            10
        );
    }

    @Serdeable
    public record CostCenter(
        @Nullable
        UUID id,
        @NotNull
        String name,
        @NotNull
        String code,
        @Nullable
        String frgnKey,
        @NotNull
        UUID userCreated,
        @Nullable
        LocalDateTime dateCreated,
        @NotNull
        UUID userUpdated,
        @Nullable
        LocalDateTime lastUpdated,
        Integer version
    ) {

        @JsonProperty("label")
        @Schema(name = "label")
        public String label() {
            return "this is label";
        }

        @JsonProperty("label")
        @Schema(name = "label")
        public String setLabel(String ll) {
            return "this is label";
        }
    }
}
