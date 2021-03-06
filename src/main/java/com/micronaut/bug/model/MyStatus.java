package com.micronaut.bug.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.HashMap;
import java.util.Map;

@RequiredArgsConstructor
@Getter
public enum MyStatus {

    ACTIVE('a'),
    BLOCKED('b'),
    ;

    public static final Map<Character, MyStatus> BY_CODE;

    static {

        var byCode = new HashMap<Character, MyStatus>();
        for (var value : values()) {
            byCode.put(value.code, value);
        }

        BY_CODE = Map.copyOf(byCode);
    }

    private final char code;

    public static MyStatus byCode(char code) {
        var value = BY_CODE.get(code);
        if (value == null) {
            throw new IllegalArgumentException("Unknown code: " + code);
        }
        return value;
    }
}
