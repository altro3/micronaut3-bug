package com.altro.common.util.api.json

import com.fasterxml.jackson.annotation.JsonInclude.Include
import tools.jackson.core.StreamWriteFeature
import tools.jackson.databind.DeserializationFeature
import tools.jackson.databind.MapperFeature
import tools.jackson.databind.PropertyNamingStrategies
import tools.jackson.databind.introspect.DefaultAccessorNamingStrategy
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.blackbird.BlackbirdModule
import tools.jackson.module.kotlin.KotlinModule

object JsonUtil {

    fun createMapper() =
        jonBuilder().build()

    fun jonBuilder() =
        JsonMapper.builder()
            .accessorNaming(
                DefaultAccessorNamingStrategy.Provider()
                    .withFirstCharAcceptance(true, true)
            )
            .changeDefaultPropertyInclusion {
                it.withValueInclusion(Include.NON_NULL)
                    .withContentInclusion(Include.NON_NULL)
            }
            .propertyNamingStrategy(PropertyNamingStrategies.LOWER_CAMEL_CASE)
            .enable(MapperFeature.USE_GETTERS_AS_SETTERS)
            .enable(StreamWriteFeature.WRITE_BIGDECIMAL_AS_PLAIN)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .addModule(BlackbirdModule())
            .addModule(KotlinModule.Builder().build())
}