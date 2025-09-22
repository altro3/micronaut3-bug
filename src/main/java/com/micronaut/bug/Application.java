package com.micronaut.bug;

import io.micronaut.context.ApplicationContextBuilder;
import io.micronaut.context.ApplicationContextConfigurer;
import io.micronaut.context.ExecutionHandleLocator;
import io.micronaut.context.annotation.ContextConfigurer;
import io.micronaut.http.HttpResponse;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.runtime.Micronaut;
import io.micronaut.web.router.DefaultRouteBuilder;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import static io.micronaut.http.HttpResponse.ok;

@ContextConfigurer
public class Application implements ApplicationContextConfigurer {

    @Override
    public void configure(ApplicationContextBuilder builder) {
        builder.deduceEnvironment(false)
            .banner(false)
            .defaultEnvironments("local");

    }

    public static void main(String[] args) {
        Micronaut.run(Application.class, args);
    }

    @Singleton
    public static class MyRoutes extends DefaultRouteBuilder {

        public MyRoutes(ExecutionHandleLocator executionHandleLocator,
                        UriNamingStrategy uriNamingStrategy) {
            super(executionHandleLocator, uriNamingStrategy);
        }

        @Inject
        void issuesRoutes() {
            GET("/my-endpoint", IssuesController.class, "handleEndpoint");
            GET("/second-endpoint", IssuesController.class, "handleEndpoint2")
                .where(rq -> rq.accept().contains(MediaType.APPLICATION_JSON_TYPE))
            ;
        }
    }
}
