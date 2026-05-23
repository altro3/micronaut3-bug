package com.altro.common.client

import org.springframework.http.HttpMethod
import org.springframework.validation.annotation.Validated

@Validated
class Endpoint(
    path: String,
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
