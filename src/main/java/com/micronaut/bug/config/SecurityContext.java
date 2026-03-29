package com.micronaut.bug.config;

/**
 * Контейнер для хранения данных пользователя в рамках текущего потока (Scope).
 * Работает на Java 21+ с флагом --enable-preview.
 */
public class SecurityContext {

    /**
     * ScopedValue — неизменяемое значение, которое "живет" только внутри 
     * выполнения определенного блока кода (фильтра/метода).
     */
    public static final ScopedValue<User> CURRENT_USER = ScopedValue.newInstance();

    /**
     * Утилитный метод для безопасного получения пользователя.
     * @return Объект User или null, если значение не привязано.
     */
    public static User getUser() {
        return CURRENT_USER.isBound() ? CURRENT_USER.get() : null;
    }
}