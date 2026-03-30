package com.micronaut.bug.config;

public class SecurityContext {

    private static final ThreadLocal<User> HOLDER = new ThreadLocal<>();

    public static void set(User user) {
        HOLDER.set(user);
    }

    public static User getUser() {
        return HOLDER.get();
    }

    public static void clear() {
        HOLDER.remove();
    }
}