package com.micronaut.bug.config.data;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import io.micronaut.core.annotation.ReflectiveAccess;

import java.util.HashMap;
import java.util.Map;

public enum VariantEnum {

    VAR1(1),
    VAR2(2),
    ;

    public static final Map<Byte, VariantEnum> BY_ID;

    static {

        var byId = new HashMap<Byte, VariantEnum>();
        for (var value : values()) {
            byId.put(value.id, value);
        }

        BY_ID = Map.copyOf(byId);
    }

    private final byte id;

    VariantEnum(int id) {
        this.id = (byte) id;
    }

    @JsonValue
    public byte getId() {
        return id;
    }

    @ReflectiveAccess
    @JsonCreator
    public static VariantEnum byId(byte id) {
        var value = BY_ID.get(id);
        if (value == null) {
            throw new IllegalArgumentException("Unknown id: " + id);
        }
        return value;
    }

}
