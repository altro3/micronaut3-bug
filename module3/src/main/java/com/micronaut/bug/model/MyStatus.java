package com.micronaut.bug.model;

import java.util.HashMap;
import java.util.Map;

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

    MyStatus(char code) {
        this.code = code;
    }

    public char getCode() {
        return code;
    }

    public static MyStatus byCode(char code) {
        var value = BY_CODE.get(code);
        if (value == null) {
            throw new IllegalArgumentException("Unknown code: " + code);
        }
        return value;
    }
}
