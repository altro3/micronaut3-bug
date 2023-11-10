package com.micronaut.bug;

import io.micronaut.context.ApplicationContextBuilder;
import io.micronaut.context.ApplicationContextConfigurer;
import io.micronaut.context.annotation.ContextConfigurer;
import io.micronaut.runtime.Micronaut;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;

@OpenAPIDefinition(
        info = @Info(
                title = "broken-micronaut-openapi-expand",
                version = "sdsd",
                description = ""
        )
)
@ContextConfigurer
public class Application implements ApplicationContextConfigurer {

    @Override
    public void configure(ApplicationContextBuilder builder) {
        System.out.println("Java configurer loaded");
        builder.deduceEnvironment(false)
                .banner(false)
                .eagerInitConfiguration(true)
                .eagerInitSingletons(true)
                .defaultEnvironments("local");
    }

    public static void main(String[] args) {
        Micronaut.run(Application.class, args);
    }

}
