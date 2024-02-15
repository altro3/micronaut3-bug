package com.micronaut.bug

import io.micronaut.runtime.Micronaut

object Application {

    @JvmStatic
    fun main(args: Array<String>) {
        Micronaut.build(*args)
                .defaultEnvironments("local")
                .banner(false)
                .start()
    }
}
