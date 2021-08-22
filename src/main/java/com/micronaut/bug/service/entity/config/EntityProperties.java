package com.micronaut.bug.service.entity.config;

import io.micronaut.context.annotation.ConfigurationProperties;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.PostConstruct;
import java.time.Duration;

@Slf4j
@Getter
@Setter
@ConfigurationProperties("app.entity")
public class EntityProperties {

    private String prop;

    /**
     * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     * That is PROBLEM block. If you comment this - all works fine
     */
    @PostConstruct
    public void init() {
        log.info("prop = {}", prop);
    }
}
