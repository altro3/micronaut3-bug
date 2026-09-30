package com.altro.servicekora.api.dto

import io.koraframework.json.common.annotation.Json
import io.koraframework.validation.common.annotation.NotEmpty
import io.koraframework.validation.common.annotation.Pattern
import io.koraframework.validation.common.annotation.Size
import io.koraframework.validation.common.annotation.Valid

@Json
@Valid
data class CreateUserRequest(
    @Size(min = 2, max = 50)
    val name: String,
    @NotEmpty
    @Pattern("^[A-Za-z0-9+_.-]+@(.+)$")
    val email: String,
)
