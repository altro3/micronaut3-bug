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

    @Bean
    public TypeConverter<String, EnumParam> enumConverter(ObjectMapper objectMapper) {
        return (value, targetType, context) -> {
            try {
                return Optional.of(objectMapper.readValue(value, EnumParam.class));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }
}
