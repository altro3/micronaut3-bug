package com.micronaut.bug.config;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.micronaut.bug.config.data.ConfigData;
import com.micronaut.bug.config.data.VariantEnum;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.util.StringUtils;
import jakarta.inject.Named;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.PostConstruct;
import java.util.EnumMap;
import java.util.Map;

@Slf4j
@Factory
public class MyEntityConfig {

    @Bean
    @Named("myConfigs")
    public Map<VariantEnum, ConfigData> myConfigs(ObjectMapper objectMapper) {

        var myMapper = objectMapper.copy()
                .enable(JsonParser.Feature.ALLOW_COMMENTS);

        var myConfigs = new EnumMap<VariantEnum, ConfigData>(VariantEnum.class);

        for (var variant : VariantEnum.values()) {
            try {
                var fileUrl = ClassLoader.getSystemResource("configs/" + variant.code + ".json");
                if (fileUrl == null) {
                    continue;
                }
                var configData = myMapper.readValue(fileUrl, ConfigData.class);
                log.info("Read {}", configData);
                myConfigs.put(variant, configData);
            } catch (Throwable t) {
                log.error("Can't read config for variant {}", variant, t);
            }
        }

        return myConfigs;
    }


    @PostConstruct
    public void init() {
        System.setProperty("system.property", StringUtils.EMPTY_STRING);
    }
}
