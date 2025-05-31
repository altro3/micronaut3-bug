package com.micronaut.bug.service;

import com.micronaut.bug.api.EnumParam;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.convert.TypeConverter;
import io.micronaut.serde.ObjectMapper;

import java.io.IOException;

@Factory
public class EnumConverterConfig {

    private final ObjectMapper objectMapper;

    public EnumConverterConfig(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Bean
    public TypeConverter<String, EnumParam> strToEnumConverter() {
        return commonToEnumConverter(EnumParam.class, objectMapper);
    }

    @Bean
    public TypeConverter<EnumParam, String> enumToStrConverter() {
        return commonToStrConverter(EnumParam.class, objectMapper);
    }

    public static <T> TypeConverter<T, String> commonToStrConverter(Class<T> clazz, ObjectMapper objectMapper) {
        return TypeConverter.of(clazz, String.class, (value) -> {
            try {
                return objectMapper.writeValueAsString(value);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    public static <T> TypeConverter<String, T> commonToEnumConverter(Class<T> clazz, ObjectMapper objectMapper) {
        return TypeConverter.of(String.class, clazz, (value) -> {
            try {
                return objectMapper.readValue(value, clazz);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
