package com.micronaut.bug.client

import jakarta.validation.constraints.NotNull
import org.springframework.http.HttpMethod
import org.springframework.validation.annotation.Validated

/**
 * Удобно использовать для описания ендпоинтов в конфигурациях.
 */
@Validated
class Endpoint(
    @NotNull
    path: String,
    @field:NotNull
    val method: HttpMethod,
    val withApiKey: Boolean = true,
) {

    val path: String = normalize(path)

    companion object {

        private fun normalize(path: String): String {
            return if (path.startsWith("http://") || path.startsWith("https://")) {
                path
            } else if (path.isNotEmpty() && !path.startsWith('/')) {
                "/$path"
            } else {
                path
            }
        }
    }
}
