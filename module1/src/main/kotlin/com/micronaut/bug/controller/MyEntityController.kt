package com.micronaut.bug.controller

import io.micronaut.core.annotation.Nullable
import io.micronaut.serde.annotation.Serdeable
import io.swagger.v3.oas.annotations.media.Schema
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.math.BigDecimal

@RestController
class MyEntityController {

    @GetMapping("/test")
    fun test(@RequestParam(required = false) param: String?): Bird {
        return Bird(
            numWings = 2,
            beakLength = BigDecimal.valueOf(12, 1),
            featherDescription = "Large blue and white feathers"
        )
    }

    @Serdeable
    data class Bird(
        @Nullable
        @Schema(name = "numWings", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        var numWings: Int? = null,
        @Nullable
        @Schema(name = "beakLength", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        var beakLength: BigDecimal? = null,
        @Nullable
        @Schema(name = "featherDescription", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        var featherDescription: String? = null,
    )

}
