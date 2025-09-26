package com.micronaut.bug.controller

import io.micronaut.context.annotation.Bean
import io.micronaut.context.annotation.Factory
import io.micronaut.core.convert.TypeConverter
import java.util.Optional

@Factory
class EnumConverterConfig {

    @Bean
    fun strToEnumConverter() = TypeConverter<String?, EnumParam?> { v, _, _ -> Optional.of(EnumParam.fromValue(v.toInt())) }

    @Bean
    fun enumToStrConverter() = TypeConverter<EnumParam?, String?> { v, _, _ -> Optional.of(v.value.toString()) }
}
