package com.micronaut.bug.service;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Map;

import io.micronaut.serde.ObjectMapper;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Named;
import jakarta.inject.Singleton;
import lombok.extern.slf4j.Slf4j;

import com.micronaut.bug.api.Bird;
import com.micronaut.bug.api.ColorEnum;
import com.micronaut.bug.api.Mammal;
import com.micronaut.bug.api.Reptile;
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

        var bird = new Bird()
                .numWings(2)
                .beakLength(BigDecimal.valueOf(12, 1))
                .featherDescription("Large blue and white feathers")
                .color(ColorEnum.BLUE);

        var mammal = new Mammal(20.5f, "A typical Canadian beaver")
                .color(ColorEnum.BLUE);

        var reptile = new Reptile(0, true)
                .fangDescription("A pair of venomous fangs")
                .color(ColorEnum.BLUE);

        log.info("Bird: {}", bird);
        log.info("Mammal: {}", mammal);
        log.info("Reptile: {}", reptile);

        try {

            var birdStr = "";
            var mammalStr = "";
            var reptileStr = "";

            log.info("Bird json: {}", birdStr = mapper.writeValueAsString(bird));
            log.info("Mammal json: {}", mammalStr = mapper.writeValueAsString(mammal));
            log.info("Reptile json: {}", reptileStr = mapper.writeValueAsString(reptile));

            log.info("Bird deserialized: {}", mapper.readValue(birdStr, Bird.class));
            log.info("Mammal deserialized: {}", mapper.readValue(mammalStr, Mammal.class));
            log.info("Reptile deserialized: {}", mapper.readValue(reptileStr, Reptile.class));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
