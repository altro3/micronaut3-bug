package com.micronaut.bug.service.entity.service;

import com.micronaut.bug.service.entity.config.EntityProperties;
import io.micronaut.runtime.event.annotation.EventListener;
import io.micronaut.runtime.server.event.ServerStartupEvent;
import jakarta.inject.Singleton;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Singleton
public class EntityService {

    private final EntityProperties entityProperties;

    @EventListener
    public void init(ServerStartupEvent startupEvent) {
        loadFromDb();
    }

    public void loadFromDb() {

        log.info("Test property: {}", entityProperties.getProp());

    }
}
