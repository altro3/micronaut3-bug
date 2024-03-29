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

import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;

import io.micronaut.core.annotation.Nullable;
import io.micronaut.serde.annotation.Serdeable;
import io.swagger.v3.oas.annotations.media.Schema;

import jakarta.annotation.Generated;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * Bird
 */
@Serdeable
@JsonPropertyOrder({
    Bird.JSON_PROPERTY_NUM_WINGS,
    Bird.JSON_PROPERTY_BEAK_LENGTH,
    Bird.JSON_PROPERTY_FEATHER_DESCRIPTION
})
@Generated("io.micronaut.openapi.generator.JavaMicronautServerCodegen")
public class Bird extends Animal {

    public static final String JSON_PROPERTY_NUM_WINGS = "numWings";
    public static final String JSON_PROPERTY_BEAK_LENGTH = "beakLength";
    public static final String JSON_PROPERTY_FEATHER_DESCRIPTION = "featherDescription";

    @Nullable
    @Schema(name = "numWings", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
    @JsonProperty(JSON_PROPERTY_NUM_WINGS)
    @JsonInclude(JsonInclude.Include.USE_DEFAULTS)
    private Integer numWings;

    @Nullable
    @Schema(name = "beakLength", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
    @JsonProperty(JSON_PROPERTY_BEAK_LENGTH)
    @JsonInclude(JsonInclude.Include.USE_DEFAULTS)
    private BigDecimal beakLength;

    @Nullable
    @Schema(name = "featherDescription", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
    @JsonProperty(JSON_PROPERTY_FEATHER_DESCRIPTION)
    @JsonInclude(JsonInclude.Include.USE_DEFAULTS)
    private String featherDescription;

    /**
     * @return the numWings property value
     */
    public Integer getNumWings() {
        return numWings;
    }

    /**
     * @return the numWings property value wrapped in an optional
     */
    @JsonIgnore
    public Optional<Integer> getNumWingsOptional() {
        return Optional.ofNullable(numWings);
    }

    /**
     * Set the numWings property value
     */
    public void setNumWings(Integer numWings) {
        this.numWings = numWings;
    }

    /**
     * Set numWings in a chainable fashion.
     *
     * @return The same instance of Bird for chaining.
     */
    public Bird numWings(Integer numWings) {
        this.numWings = numWings;
        return this;
    }

    /**
     * @return the beakLength property value
     */
    public BigDecimal getBeakLength() {
        return beakLength;
    }

    /**
     * @return the beakLength property value wrapped in an optional
     */
    @JsonIgnore
    public Optional<BigDecimal> getBeakLengthOptional() {
        return Optional.ofNullable(beakLength);
    }

    /**
     * Set the beakLength property value
     */
    public void setBeakLength(BigDecimal beakLength) {
        this.beakLength = beakLength;
    }

    /**
     * Set beakLength in a chainable fashion.
     *
     * @return The same instance of Bird for chaining.
     */
    public Bird beakLength(BigDecimal beakLength) {
        this.beakLength = beakLength;
        return this;
    }

    /**
     * @return the featherDescription property value
     */
    public String getFeatherDescription() {
        return featherDescription;
    }

    /**
     * @return the featherDescription property value wrapped in an optional
     */
    @JsonIgnore
    public Optional<String> getFeatherDescriptionOptional() {
        return Optional.ofNullable(featherDescription);
    }

    /**
     * Set the featherDescription property value
     */
    public void setFeatherDescription(String featherDescription) {
        this.featherDescription = featherDescription;
    }

    /**
     * Set featherDescription in a chainable fashion.
     *
     * @return The same instance of Bird for chaining.
     */
    public Bird featherDescription(String featherDescription) {
        this.featherDescription = featherDescription;
        return this;
    }

    @Override
    public Bird propertyClass(String propertyClass) {
        super.setPropertyClass(propertyClass);
        return this;
    }

    @Override
    public Bird color(ColorEnum color) {
        super.setColor(color);
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
        Bird bird = (Bird) o;
        return Objects.equals(numWings, bird.numWings) &&
            Objects.equals(beakLength, bird.beakLength) &&
            Objects.equals(featherDescription, bird.featherDescription) &&
            super.equals(o);
    }

    @Override
    public int hashCode() {
        return Objects.hash(numWings, beakLength, featherDescription, super.hashCode());
    }

    @Override
    public String toString() {
        return "Bird("
            + "numWings: " + getNumWings() + ", "
            + "beakLength: " + getBeakLength() + ", "
            + "featherDescription: " + getFeatherDescription() + ", "
            + "propertyClass: " + getPropertyClass() + ", "
            + "color: " + getColor()
            + ")";
    }

}
