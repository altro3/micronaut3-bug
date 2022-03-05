package com.micronaut.bug.config;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.fasterxml.jackson.module.paramnames.ParameterNamesModule;
import com.micronaut.bug.config.data.ConfigData;
import com.micronaut.bug.config.data.VariantEnum;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.util.StringUtils;
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
    public Map<VariantEnum, ConfigData> myConfigs() {

        var mapper = JsonMapper.builder()
                .defaultPropertyInclusion(JsonInclude.Value.construct(JsonInclude.Include.NON_NULL, JsonInclude.Include.NON_NULL))
                .defaultDateFormat(StdDateFormat.getDateTimeInstance())
                .propertyNamingStrategy(PropertyNamingStrategies.LOWER_CAMEL_CASE)
                .visibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY)
                .visibility(PropertyAccessor.CREATOR, JsonAutoDetect.Visibility.ANY)
                .defaultLocale(Locale.US)
                .enable(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS)
                .enable(JsonParser.Feature.ALLOW_COMMENTS)
                .disable(
                        SerializationFeature.INDENT_OUTPUT,
                        SerializationFeature.WRITE_DATE_TIMESTAMPS_AS_NANOSECONDS,
                        SerializationFeature.WRITE_DATES_AS_TIMESTAMPS
                )
                .disable(
                        DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                        DeserializationFeature.READ_DATE_TIMESTAMPS_AS_NANOSECONDS
                )
                .addModule(new JavaTimeModule())
                .addModule(new ParameterNamesModule(JsonCreator.Mode.PROPERTIES))
                .build();

        var myConfigs = new EnumMap<VariantEnum, ConfigData>(VariantEnum.class);

        for (var variant : VariantEnum.values()) {
            try {
                var fileUrl = ClassLoader.getSystemResource("configs/" + variant.code + ".json");
                if (fileUrl == null) {
                    continue;
                }
                var configData = mapper.readValue(fileUrl, ConfigData.class);
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
