package com.altro.common.util.api.json

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.PropertyNamingStrategies
import com.fasterxml.jackson.databind.json.JsonMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.blackbird.BlackbirdModule
import com.fasterxml.jackson.module.kotlin.KotlinModule

object JsonUtil {

    fun createMapper() =
        jonBuilder().build()

    fun jonBuilder() =
        JsonMapper.builder()
            .defaultPropertyInclusion(JsonInclude.Value.ALL_NON_NULL)
            .propertyNamingStrategy(PropertyNamingStrategies.LOWER_CAMEL_CASE)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .addModule(JavaTimeModule())
            .addModule(BlackbirdModule())
            .addModule(KotlinModule.Builder().build())
}