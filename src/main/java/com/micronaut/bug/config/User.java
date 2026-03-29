package com.micronaut.bug.config;

public record User(
    String id,
    String username,
    String role
) {
}