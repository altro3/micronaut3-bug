package com.micronaut.bug.controller

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonValue
import io.micronaut.core.annotation.Introspected

@Introspected
enum class EnumParam(
    @get:JsonValue
    val value: Int
) {

    VALUE_1(10),
    VALUE_2(20),
    ;

    companion object {

        @JvmField
        val VALUE_MAPPING = entries.associateBy { it.value }

        @JsonCreator
        @JvmStatic
        fun fromValue(value: Int?): EnumParam {
            return VALUE_MAPPING[value]!!
        }
    }
}
