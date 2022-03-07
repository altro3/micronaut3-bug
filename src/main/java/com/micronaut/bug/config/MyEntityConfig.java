package com.micronaut.bug.config;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.fasterxml.jackson.module.paramnames.ParameterNamesModule;
import com.micronaut.bug.config.data.ConfigData;
import com.micronaut.bug.config.data.VariantEnum;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.util.StringUtils;
import io.micronaut.jackson.modules.BeanIntrospectionModule;
import jakarta.inject.Named;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.PostConstruct;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;

@Slf4j
@Factory
public class MyEntityConfig {

    @Bean
    @Named("myConfigs")
    public Map<VariantEnum, ConfigData> myConfigs(ObjectMapper objectMapper) {

        var myMapper = objectMapper.copy();

        myMapper.enable(JsonParser.Feature.ALLOW_COMMENTS);

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
