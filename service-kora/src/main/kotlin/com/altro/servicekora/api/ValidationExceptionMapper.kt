package com.altro.servicekora.api

import io.koraframework.common.annotation.Component
import io.koraframework.http.common.body.HttpBodyOutput
import io.koraframework.http.server.common.request.HttpServerRequest
import io.koraframework.http.server.common.response.HttpServerResponse
import io.koraframework.http.server.common.response.HttpServerResponseMapper
import io.micrometer.core.instrument.config.validate.ValidationException

@Component
class ValidationExceptionMapper : HttpServerResponseMapper<ValidationException> {

    override fun apply(request: HttpServerRequest?, result: ValidationException?): HttpServerResponse? {
        val message = result?.message ?: "Validation failed"
        val body = message.toByteArray(Charsets.UTF_8)
        return HttpServerResponse.of(400, HttpBodyOutput.of("plain/text") { it.write(body) })
    }
}
