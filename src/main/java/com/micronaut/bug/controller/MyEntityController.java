package com.micronaut.bug.controller;

import com.micronaut.bug.service.MyEntityService;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.Head;
import io.micronaut.http.annotation.Header;
import io.micronaut.scheduling.TaskExecutors;
import io.micronaut.scheduling.annotation.ExecuteOn;
import jakarta.validation.constraints.NotNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Optional;
import java.util.OptionalInt;

@ExecuteOn(TaskExecutors.BLOCKING)
@Slf4j
@Controller
public class MyEntityController {

    private final MyEntityService entityService;

    public MyEntityController(MyEntityService entityService) {
        this.entityService = entityService;
    }

    @Head("/")
    void fileAvailable(
        @Header("X-TEST-crc32") @NotNull String xtESTCrc32,
        @Header("X-TEST-sha512") @NotNull String xtESTSha512,
        @Header("X-TEST-size") @NotNull Long xtESTSize
    ) {
        log.info("xtESTCrc32: {}", xtESTCrc32);
    }

    @Get("/test")
    public Optional<Integer> test() {
        return Optional.of(100);
    }

    @Get("/test2")
    public OptionalInt test2() {
        return OptionalInt.of(100);
    }
}
