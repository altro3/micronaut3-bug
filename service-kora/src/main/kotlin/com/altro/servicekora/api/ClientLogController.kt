package com.altro.servicekora.api

import com.altro.servicekora.api.dto.UserRq
import com.altro.servicekora.api.dto.UserRs
import io.github.oshai.kotlinlogging.KotlinLogging
import io.koraframework.common.annotation.Component
import io.koraframework.http.common.annotation.HttpRoute
import io.koraframework.http.server.common.annotation.HttpController
import io.koraframework.json.common.JsonReader
import io.koraframework.json.common.annotation.Json
import java.net.URI
import java.time.LocalDateTime
import java.util.UUID

@Component
@HttpController
class ClientLogController {

    private val log = KotlinLogging.logger {}
    private val uriReader = JsonReader<URI> { parser -> URI.create(parser.string) }


    @HttpRoute(method = "post", path = "/route")
    @Json
    fun testRoute(@Json rq: UserRq): UserRs {
        log.info { "Received createUser request: name=${rq.name}, email=${rq.email}" }
        return UserRs(
            id = UUID.randomUUID().toString(),
            name = rq.name,
            email = rq.email,
            createdAt = LocalDateTime.now(),
        )
    }
}
