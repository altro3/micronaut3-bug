package com.micronaut.bug.service;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Map;

import io.micronaut.serde.ObjectMapper;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Named;
import jakarta.inject.Singleton;
import lombok.extern.slf4j.Slf4j;

import com.micronaut.bug.config.MyEntityProperties;
import com.micronaut.bug.config.data.ConfigData;
import com.micronaut.bug.config.data.VariantEnum;
import com.micronaut.bug.dao.MyEntityDao;
import com.micronaut.bug.model.MyEntity;

@Slf4j
@Singleton
public class MyEntityService {

    private final MyEntityProperties entityProperties;
    private final MyEntityDao myEntityDao;
    private final Map<VariantEnum, ConfigData> myConfigs;

    public MyEntityService(MyEntityProperties entityProperties, MyEntityDao myEntityDao,
                           @Named("myConfigs") Map<VariantEnum, ConfigData> myConfigs) {
        this.entityProperties = entityProperties;
        this.myEntityDao = myEntityDao;
        this.myConfigs = myConfigs;
    }

    @PostConstruct
    public void init() {

        save();

        loadFromDb();
    }

    public void process() {

        var mapper = ObjectMapper.getDefault();
    }

    public MyEntity save() {
/*
        var myEntity = myEntityDao.save(MyEntity.builder()
                .field("field")
                .jsonField(JsonField.builder()
                        .id(12)
                        .text(" this is text")
                        .number(1233564565646L)
                        .build())
                .jsonFieldBin(JsonField.builder()
                        .id(12)
                        .text(" this is text")
                        .number(1233564565646L)
                        .build())
                .status(MyStatus.ACTIVE).build());
*/

//        log.info("Entity {} created", myEntity);

        return null;
    }

    public void loadFromDb() {

//        log.info("Test property: {}", entityProperties.getProp());

        var entities = myEntityDao.findAll();

//        log.info("Found {} entities", entities);

    }
}
