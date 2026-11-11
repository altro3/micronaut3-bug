package com.micronaut.bug;

import io.micronaut.context.ApplicationContextBuilder;
import io.micronaut.context.ApplicationContextConfigurer;
import io.micronaut.context.ExecutionHandleLocator;
import io.micronaut.context.annotation.ContextConfigurer;
import io.micronaut.http.MediaType;
import io.micronaut.runtime.Micronaut;
import io.micronaut.web.router.DefaultRouteBuilder;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import static io.micronaut.http.HttpResponse.ok;

@ContextConfigurer
public class App implements ApplicationContextConfigurer {

    @Override
    public void configure(ApplicationContextBuilder builder) {
        builder.deduceEnvironment(false)
            .banner(false)
            .defaultEnvironments("local");

    }

    static void main(String[] args) {
        Micronaut.run(App.class, args);
    }
}
