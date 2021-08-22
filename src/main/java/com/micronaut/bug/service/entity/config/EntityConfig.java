package com.micronaut.bug.service.entity.config;

import io.micronaut.context.annotation.Factory;
import io.micronaut.core.util.StringUtils;
import lombok.RequiredArgsConstructor;

import javax.annotation.PostConstruct;

@RequiredArgsConstructor
@Factory
public class EntityConfig {

    private final EntityProperties entityProperties;

    @PostConstruct
    public void init() {
        System.setProperty("system.property", StringUtils.EMPTY_STRING);

    }
}
