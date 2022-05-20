package com.micronaut.bug.controller.openapi;

import io.micronaut.http.HttpResponse;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Post;
import io.micronaut.http.multipart.CompletedFileUpload;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Encoding;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.validation.constraints.NotNull;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import java.util.Map;

@Path("/path/{input}")
@Controller("/path/{input}")
public class OpenApiController {

    @Post(consumes = MediaType.MULTIPART_FORM_DATA)
    @Operation(summary = "Endpoint summary", description = "Endpoint description", tags = "Print",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Bad document", content = @Content(mediaType = MediaType.ALL, schema = @Schema(type = "blob", format = "binary"))),
                    @ApiResponse(responseCode = "400", description = "Invalid template", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
                    @ApiResponse(responseCode = "406", description = "Invalid format", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
                    @ApiResponse(responseCode = "500", description = "Server error", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
            })
    @POST
    public Mono<HttpResponse<byte[]>> processSync(@Parameter(required = true, name = "input", in = ParameterIn.PATH, schema = @Schema(implementation = DocumentFormat.class)) @NotNull DocumentFormat input,
                                                  @RequestBody(required = true, description = "Template as file", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA, encoding = @Encoding(name = "template", contentType = MediaType.APPLICATION_OCTET_STREAM), schema = @Schema(implementation = UploadPrint.class))) @NotNull Flux<CompletedFileUpload> template) {
        return Mono.empty();
    }

    public enum DocumentFormat {
        HTML,
        PDF,
        DOCX
    }

    @Schema(requiredProperties = {"parameters", "template"})
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class UploadPrint {

        @Schema(name = "template", description = "Template as file", type = "object", format = "binary", required = true)
        private String template;
        @Schema(implementation = PrintParameters.class)
        private Map<String, Object> parameters;

        @Data
        @AllArgsConstructor
        @NoArgsConstructor
        public static class PrintParameters {

            @Schema
            private Map<String, Object> parameters;
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class ErrorResponse {

        private String code;
        private String message;
    }

}
