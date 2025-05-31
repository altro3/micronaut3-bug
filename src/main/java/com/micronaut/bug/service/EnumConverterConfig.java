package com.micronaut.bug.service;

import com.micronaut.bug.api.EnumParam;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.convert.TypeConverter;
import io.micronaut.serde.ObjectMapper;

import java.io.IOException;
import java.util.Optional;

@Factory
public class EnumConverterConfig {

    private final ObjectMapper objectMapper;

    public EnumConverterConfig(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Bean
    public TypeConverter<String, EnumParam> strToEnumConverter() {
        return commonConverter(EnumParam.class, objectMapper);
    }

    @Bean
    public TypeConverter<EnumParam, String> enumToStrConverter() {
        return commonConverter(objectMapper);
    }

    public static <T> TypeConverter<T, String> commonConverter(ObjectMapper objectMapper) {
        return (value, targetType, context) -> {
            try {
                return Optional.of(objectMapper.writeValueAsString(value));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }

    public static <T> TypeConverter<String, T> commonConverter(Class<T> clazz, ObjectMapper objectMapper) {
        return (value, targetType, context) -> {
            try {
                return Optional.of(objectMapper.readValue(value, clazz));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }
}
