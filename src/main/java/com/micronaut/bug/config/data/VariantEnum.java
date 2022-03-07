package com.micronaut.bug.config.data;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import io.micronaut.core.annotation.ReflectiveAccess;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

@ReflectiveAccess
@Getter
public enum VariantEnum {

    VAR1(1, "var1"),
    VAR2(2, "var2"),
    ;

    public static final Map<Byte, VariantEnum> BY_ID;
    public static final Map<String, VariantEnum> BY_CODE;

    static {

        var byId = new HashMap<Byte, VariantEnum>();
        var byCode = new HashMap<String, VariantEnum>();
        for (var value : values()) {
            byId.put(value.id, value);
            byCode.put(value.code, value);
        }

        BY_ID = Map.copyOf(byId);
        BY_CODE = Map.copyOf(byCode);
    }

    @Getter(onMethod_ = @JsonValue)
    public final byte id;
    public final String code;

    VariantEnum(int id, String code) {
        this.id = (byte) id;
        this.code = code;
    }

    public static VariantEnum byCode(String code) {
        var value = BY_CODE.get(code);
        if (value == null) {
            throw new IllegalArgumentException("Unknown code: " + code);
        }
        return value;
    }

    @JsonCreator(mode = JsonCreator.Mode.DELEGATING)
    public static VariantEnum byId(byte id) {
        var value = BY_ID.get(id);
        if (value == null) {
            throw new IllegalArgumentException("Unknown id: " + id);
        }
        return value;
    }

}
