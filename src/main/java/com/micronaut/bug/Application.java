package com.micronaut.bug;

import io.micronaut.context.ApplicationContextBuilder;
import io.micronaut.context.ApplicationContextConfigurer;
import io.micronaut.context.ExecutionHandleLocator;
import io.micronaut.context.annotation.ContextConfigurer;
import io.micronaut.runtime.Micronaut;
import io.micronaut.web.router.DefaultRouteBuilder;
import io.micronaut.web.router.RouteBuilder;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

@OpenAPIDefinition(
        info = @Info(
                title = "${app.entity.prop}",
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
                .defaultEnvironments("local");

    }

    public static void main(String[] args) {
        var sss = Micronaut.run(Application.class, args);

        sss.getBeansOfType(RouteBuilder.class).
    }

    @Singleton
    public class MyRoutes extends DefaultRouteBuilder { // (1)

        public MyRoutes(ExecutionHandleLocator executionHandleLocator,
                        UriNamingStrategy uriNamingStrategy) {
            super(executionHandleLocator, uriNamingStrategy);
        }

        @Inject
        void issuesRoutes() { // (2)
            GET("/issues/show/{number}", (a, b) -> {}); // (3)
        }
    }
}
