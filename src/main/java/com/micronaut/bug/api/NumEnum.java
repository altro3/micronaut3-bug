/*
 * Compute API
 * API for the Compute Service
 *
 * The version of the OpenAPI document: 1.0.0
 *
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */
package com.micronaut.bug.api;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import io.micronaut.serde.annotation.Serdeable;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

/**
 * Gets or Sets ColorEnum
 */
@Serdeable
public enum NumEnum {

    RED(500),
    BLUE(1000),
    GREEN(789),
    ;

    public final static Map<Integer, NumEnum> VALUE_MAPPING = Collections.unmodifiableMap(Arrays.stream(values())
            .collect(Collectors.toMap(v -> v.value, Function.identity())));

    private final int value;

    NumEnum(int value) {
        this.value = value;
    }

    /**
     * @return The value represented by this enum
     */
    @JsonValue
    public int getValue() {
        return value;
    }

    /**
     * Create this enum from a value.
     *
     * @return The enum
     */
    @JsonCreator
    public static NumEnum fromValue(Integer value) {
        if (!VALUE_MAPPING.containsKey(value)) {
            throw new IllegalArgumentException("Unexpected value '" + value + "'");
        }
        return VALUE_MAPPING.get(value);
    }

    @Override
    public String toString() {
        return String.valueOf(value);
    }
}
