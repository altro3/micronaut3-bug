package com.micronaut.bug.api;

import io.micronaut.serde.annotation.Serdeable;

import lombok.Data;

import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@Serdeable
public class SimpleDto {

    @JsonProperty("class")
    private String classProp;
    private Integer numProp;
    private ColorEnum color;

    private NumEnum num;
}
