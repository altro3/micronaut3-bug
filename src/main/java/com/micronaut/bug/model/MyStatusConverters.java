package com.micronaut.bug.model;

//@Factory
public class MyStatusConverters {

/*
    @Singleton
    TypeConverter<MyStatus, String> userStatusToCharConverter() {
        return (object, targetType, context) -> Optional.of(object == null ? MyStatus.BLOCKED.getCode() : object.getCode());
    }

    @Singleton
    TypeConverter<String, MyStatus> charToUserStatusConverter() {
        return (object, targetType, context) -> Optional.of(MyStatus.byCode(object));
    }
*/
}
