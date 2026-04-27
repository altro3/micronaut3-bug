package com.micronaut.bug.controller

import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class TracingController {

    @GetMapping("/trace/ping")
    fun ping() = "pong"
}