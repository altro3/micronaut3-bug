/*
 * Compute API
 * API for the Compute Service
 *
 * The version of the OpenAPI document: 1.0.0
 *
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */
package com.micronaut.bug.api;

import java.util.Objects;
import java.util.Optional;

import io.micronaut.core.annotation.Nullable;
import io.micronaut.serde.annotation.Serdeable;
import io.swagger.v3.oas.annotations.media.Schema;

import jakarta.annotation.Generated;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/**
 * Animal
 */
@Serdeable
@JsonPropertyOrder({
    Animal.JSON_PROPERTY_PROPERTY_CLASS,
    Animal.JSON_PROPERTY_COLOR
})
@Generated("io.micronaut.openapi.generator.JavaMicronautServerCodegen")
@JsonIgnoreProperties(
        value = "class", // ignore manually set class, it will be automatically generated by Jackson during serialization
        allowSetters = true // allows the class to be set during deserialization
)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "class", visible = true)
@JsonSubTypes({
    @JsonSubTypes.Type(value = Bird.class, name = "ave"),
    @JsonSubTypes.Type(value = Mammal.class, name = "mammalia"),
    @JsonSubTypes.Type(value = Reptile.class, name = "reptilia")
})
public class Animal {

    public static final String JSON_PROPERTY_PROPERTY_CLASS = "class";
    public static final String JSON_PROPERTY_COLOR = "color";

    @Nullable
    @Schema(name = "class", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
    @JsonProperty(JSON_PROPERTY_PROPERTY_CLASS)
    @JsonInclude(JsonInclude.Include.USE_DEFAULTS)
    protected String propertyClass;

    @Nullable
    @Schema(name = "color", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
    @JsonProperty(JSON_PROPERTY_COLOR)
    @JsonInclude(JsonInclude.Include.USE_DEFAULTS)
    private ColorEnum color;

    /**
     * @return the propertyClass property value
     */
    public String getPropertyClass() {
        return propertyClass;
    }

    /**
     * @return the propertyClass property value wrapped in an optional
     */
    @JsonIgnore
    public Optional<String> getPropertyClassOptional() {
        return Optional.ofNullable(propertyClass);
    }

    /**
     * Set the propertyClass property value
     */
    public void setPropertyClass(String propertyClass) {
        this.propertyClass = propertyClass;
    }

    /**
     * Set propertyClass in a chainable fashion.
     *
     * @return The same instance of Animal for chaining.
     */
    public Animal propertyClass(String propertyClass) {
        this.propertyClass = propertyClass;
        return this;
    }

    /**
     * @return the color property value
     */
    public ColorEnum getColor() {
        return color;
    }

    /**
     * @return the color property value wrapped in an optional
     */
    @JsonIgnore
    public Optional<ColorEnum> getColorOptional() {
        return Optional.ofNullable(color);
    }

    /**
     * Set the color property value
     */
    public void setColor(ColorEnum color) {
        this.color = color;
    }

    /**
     * Set color in a chainable fashion.
     *
     * @return The same instance of Animal for chaining.
     */
    public Animal color(ColorEnum color) {
        this.color = color;
        return this;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Animal animal = (Animal) o;
        return Objects.equals(propertyClass, animal.propertyClass) &&
            Objects.equals(color, animal.color);
    }

    @Override
    public int hashCode() {
        return Objects.hash(propertyClass, color);
    }

    @Override
    public String toString() {
        return "Animal("
            + "propertyClass: " + getPropertyClass() + ", "
            + "color: " + getColor()
            + ")";
    }

}
