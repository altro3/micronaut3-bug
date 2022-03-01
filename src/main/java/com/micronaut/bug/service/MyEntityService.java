package com.micronaut.bug.service;

import com.micronaut.bug.config.MyEntityProperties;
import com.micronaut.bug.config.YamlFileProperties;
import com.micronaut.bug.dao.MyEntityDao;
import com.micronaut.bug.model.MyEntity;
import com.micronaut.bug.model.MyStatus;
import jakarta.annotation.PostConstruct;
import jakarta.inject.Singleton;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Singleton
public class MyEntityService {

    private final MyEntityProperties entityProperties;
    private final MyEntityDao myEntityDao;
    private final YamlFileProperties yamlFileProperties;

    @PostConstruct
    public void init() {

        save();

        loadFromDb();
    }

    public List<String> getPropertiesByKey(String key) {
        return yamlFileProperties.getPropsByKey(key);
    }

    public MyEntity save() {
        var myEntity = myEntityDao.save(MyEntity.builder()
                .field("field")
                .status(MyStatus.ACTIVE).build());

        log.info("Entity {} created", myEntity);

        return myEntity;
    }

    public void loadFromDb() {

        log.info("Test property: {}", entityProperties.getProp());

        var entities = myEntityDao.findAll();

        log.info("Found {} entities", entities);

    }
}
