package com.micronaut.bug.controller

import io.micronaut.context.annotation.Bean
import io.micronaut.context.annotation.Factory
import io.micronaut.core.convert.TypeConverter
import io.micronaut.serde.ObjectMapper
import java.io.IOException

@Factory
class EnumConverterConfig(private val objectMapper: ObjectMapper) {

    @Bean
    fun strToEnumConverter(): TypeConverter<String, EnumParam> =
        commonToEnumConverter(EnumParam::class.java, objectMapper)

    @Bean
    fun enumToStrConverter(): TypeConverter<EnumParam, String> =
        commonToStrConverter(EnumParam::class.java, objectMapper)

    companion object {

        fun <T> commonToStrConverter(clazz: Class<T>, objectMapper: ObjectMapper): TypeConverter<T, String> =
            TypeConverter.of(clazz, String::class.java) {
                try {
                    objectMapper.writeValueAsString(it)
                } catch (e: IOException) {
                    throw RuntimeException(e)
                }
            }

        fun <T> commonToEnumConverter(clazz: Class<T>, objectMapper: ObjectMapper): TypeConverter<String, T> =
            TypeConverter.of(String::class.java, clazz) {
                try {
                    objectMapper.readValue(it, clazz)
                } catch (e: IOException) {
                    throw RuntimeException(e)
                }
            }
    }
}
