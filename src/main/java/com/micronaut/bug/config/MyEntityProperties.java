package com.micronaut.bug.config;

import io.micronaut.context.annotation.ConfigurationProperties;
import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Getter
@Setter
@ConfigurationProperties("app.entity")
public class MyEntityProperties {

    private String prop;

    @PostConstruct
    public void init() {
//        log.info("prop = {}", prop);
    }
}
