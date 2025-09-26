package com.micronaut.bug.service;

import com.micronaut.bug.api.EnumParam;
import io.micronaut.context.annotation.Bean;
import io.micronaut.context.annotation.Factory;
import io.micronaut.core.convert.TypeConverter;

import java.util.Optional;

@Factory
public class EnumConverterConfig {

    @Bean
    public TypeConverter<String, EnumParam> strToEnumConverter() {
        return (v, c, ctx) -> Optional.of(EnumParam.fromValue(Integer.valueOf(v)));
    }

    @Bean
    public TypeConverter<EnumParam, String> enumToStrConverter() {
        return (v, c, ctx) -> Optional.of(v.toString());
    }
}
