package com.micronaut.bug.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import io.micronaut.serde.annotation.Serdeable;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Serdeable
@Getter(onMethod = @__(@JsonValue))
@RequiredArgsConstructor
public enum EnumParam {

    VALUE_1(10),
    VALUE_2(20),
    ;

    public static final Map<Integer, EnumParam> VALUE_MAPPING = Map.copyOf(Arrays.stream(values())
        .collect(Collectors.toMap(v -> v.value, Function.identity())));

    private final int value;

    @JsonCreator
    public static EnumParam fromValue(int value) {
        return VALUE_MAPPING.get(value);
    }
}
