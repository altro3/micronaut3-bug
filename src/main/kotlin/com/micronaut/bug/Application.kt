package com.micronaut.bug

import io.micronaut.context.ApplicationContextBuilder
import io.micronaut.context.ApplicationContextConfigurer
import io.micronaut.context.annotation.ContextConfigurer
import io.micronaut.runtime.Micronaut
import io.swagger.v3.oas.annotations.OpenAPIDefinition
import io.swagger.v3.oas.annotations.info.Info

@OpenAPIDefinition(info = Info(title = "broken-micronaut-openapi-expand", version = "sdsd", description = ""))
@ContextConfigurer
class Application : ApplicationContextConfigurer {
    override fun configure(builder: ApplicationContextBuilder) {
        println("Java configurer loaded")
        builder.deduceEnvironment(false)
                .banner(false)
                .eagerInitConfiguration(true)
                .eagerInitSingletons(true)
                .defaultEnvironments("local")
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            Micronaut.run(Application::class.java, *args)
        }
    }
}
