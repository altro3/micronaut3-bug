package com.micronaut.bug.model;

import io.micronaut.context.annotation.Factory;
import io.micronaut.core.convert.TypeConverter;
import jakarta.inject.Singleton;

import java.util.Optional;

@Factory
public class MyStatusConverters {

    @Singleton
    TypeConverter<MyStatus, Character> userStatusToCharConverter() {
        return (object, targetType, context) -> Optional.of(object == null ? MyStatus.BLOCKED.getCode() : object.getCode());
    }

    @Singleton
    TypeConverter<Character, MyStatus> charToUserStatusConverter() {
        return (object, targetType, context) -> Optional.of(MyStatus.byCode(object));
    }
}
