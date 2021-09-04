package com.micronaut.bug.service;

import com.micronaut.bug.config.MyEntityProperties;
import com.micronaut.bug.dao.MyEntityDao;
import io.micronaut.runtime.event.annotation.EventListener;
import io.micronaut.runtime.server.event.ServerStartupEvent;
import jakarta.inject.Singleton;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Singleton
public class MyEntityService {

    private final MyEntityProperties entityProperties;
    private final MyEntityDao myEntityDao;

    @EventListener
    public void init(ServerStartupEvent startupEvent) {
        loadFromDb();
    }

    public void loadFromDb() {

        log.info("Test property: {}", entityProperties.getProp());

        var entities = myEntityDao.findAll();
    }
}
