package com.micronaut.bug.model;

import jakarta.inject.Singleton;

@Singleton
public class MyStatusConverter {
}/* implements AttributeConverter<MyStatus, Character> {

    @Override
    public Character convertToPersistedValue(MyStatus entityValue, ConversionContext context) {
        return entityValue == null ? MyStatus.BLOCKED.getCode() : entityValue.getCode();
    }

    @Override
    public MyStatus convertToEntityValue(Character persistedValue, ConversionContext context) {
        return MyStatus.byCode(persistedValue);
    }
}
*/
