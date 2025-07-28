package com.micronaut.bug.controller

import io.micronaut.data.model.Page
import io.micronaut.data.model.Pageable
import io.micronaut.http.annotation.*
import io.swagger.v3.oas.annotations.tags.Tag

@Controller("/api/model/something")
@Tag(name = "Something API")
class SomethingController {
    @Get
    fun getAll(pageable: Pageable): Page<Something>? = null

    @Post("id")
    fun getByIds(@Body ids: Set<String>): List<Something>? = null

    @Get("{id}")
    fun get(id: String): Something? = null

    @Post("{id}")
    fun create(id: String, @Body request: SomethingChangeRequest): Something? = null

    @Put("{id}")
    fun update(id: String, @Body request: SomethingChangeRequest): Something? = null

    @Delete("{id}")
    fun delete(id: String): Unit? = null

    class Something(
        val id: String
    )

    class SomethingChangeRequest(
        val id: String
    )
}
