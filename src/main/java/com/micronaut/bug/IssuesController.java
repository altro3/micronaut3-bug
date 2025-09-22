package com.micronaut.bug;

import io.micronaut.context.annotation.Executable;
import io.micronaut.http.HttpResponse;
import io.micronaut.http.annotation.Controller;

import static io.micronaut.http.HttpResponse.ok;

@Controller
public class IssuesController {

    @Executable
    public String handleEndpoint() {
        return "This is functional endpoint";
    }

    @Executable
    public HttpResponse<String> handleEndpoint2() {
        return ok()
            .header("my-header", "my-header-value")
            .body("{\"name\": \"value\"}");
    }
}
