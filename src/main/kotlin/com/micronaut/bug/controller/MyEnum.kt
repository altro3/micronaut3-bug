package com.micronaut.bug.controller

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonValue
import io.micronaut.serde.annotation.Serdeable
import jakarta.annotation.Generated

/**
 * Gets or Sets browseSearchOrders_apiVersion_parameter
 *
 * @param value The value represented by this enum
 */
@Serdeable
@Generated("io.micronaut.openapi.generator.KotlinMicronautClientCodegen")
enum class MyEnum(
    @get:JsonValue val value: String
) {

    @JsonProperty("v1")
    V1("v1"),

    @JsonProperty("v2")
    V2("v2"),

    @JsonProperty("v3")
    V3("v3"),

    @JsonProperty("v4")
    V4("v4"),

    @JsonProperty("v5")
    V5("v5"),

    @JsonProperty("v6")
    V6("v6"),

    @JsonProperty("v7")
    V7("v7");

    override fun toString(): String {
        return value
    }

    companion object {

        @JvmField
        val VALUE_MAPPING = entries.associateBy { it.value }

        /**
         * Create this enum from a value.
         *
         * @param value value for enum
         *
         * @return The enum
         */
        @JsonCreator
        @JvmStatic
        fun fromValue(value: String): MyEnum {
            require(VALUE_MAPPING.containsKey(value)) { "Unexpected value '$value'" }
            return VALUE_MAPPING[value]!!
        }
    }
}
