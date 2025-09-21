package com.micronaut.bug;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.function.RouterFunction;
import org.springframework.web.servlet.function.ServerResponse;

import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.web.servlet.function.RequestPredicates.GET;
import static org.springframework.web.servlet.function.RequestPredicates.accept;
import static org.springframework.web.servlet.function.RequestPredicates.headers;
import static org.springframework.web.servlet.function.RouterFunctions.route;
import static org.springframework.web.servlet.function.ServerResponse.ok;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Configuration
    public static class RouteConfig {

        @Bean
        RouterFunction<ServerResponse> myRouter() {
            return route(
                GET("/my-endpoint"),
                rq -> ok().body("This is functional endpoint")
            ).and(route(
                GET("/second-endpoint").and(accept(APPLICATION_JSON)),
                rq -> ok()
                    .headers(headers -> headers.add("my-header", "my-header-value"))
                    .body("{\"name\": \"value\"}")
            ));
        }
    }
}
