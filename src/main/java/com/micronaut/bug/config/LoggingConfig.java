package com.micronaut.bug.config;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.reactivestreams.Publisher;
import org.slf4j.MDC;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.server.reactive.ServerHttpRequestDecorator;
import org.springframework.http.server.reactive.ServerHttpResponseDecorator;
import org.springframework.web.server.WebFilter;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.context.Context;

import java.nio.charset.StandardCharsets;
import java.util.UUID;

@Slf4j
@Configuration
public class LoggingConfig {

    @Bean
    WebFilter tracingFilter() {
        return (exchange, chain) -> {
            var rq = exchange.getRequest();
            String traceId = null;
            if (rq.getHeaders().containsKey("x-req-id")) {
                traceId = rq.getHeaders().getFirst("x-req-id");
            }
            if (traceId == null) {
                traceId = UUID.randomUUID().toString().replace("-", "");
                rq.mutate().header("x-req-id", traceId);
            }
            MDC.putCloseable("x-req-id", traceId);
            return chain.filter(exchange)
                .contextWrite(Context.of("x-req-id", traceId))
                .contextWrite()
        };
    }

    @Bean
    WebFilter logFilter() {
        return (exchange, chain) -> {
            var rq = exchange.getRequest();
            return DataBufferUtils.join(rq.getBody())
                .flatMap(dataBuffer -> {

                    var bytes = new byte[dataBuffer.readableByteCount()];
                    dataBuffer.read(bytes);
                    DataBufferUtils.release(dataBuffer); // Release the buffer

                    var requestBody = new String(bytes, StandardCharsets.UTF_8);
                    log.info("!!! Request Body: {}", requestBody);
                    var rs = exchange.getResponse();

                    // Re-create the request with the logged body
                    var decoratedRq = new ServerHttpRequestDecorator(rq) {
                        @NotNull
                        @Override
                        public Flux<DataBuffer> getBody() {
                            return Flux.just(rs.bufferFactory().wrap(bytes));
                        }
                    };

                    var decoratedRs = new ServerHttpResponseDecorator(rs) {


                        @NotNull
                        @Override
                        public Mono<Void> writeWith(@NotNull Publisher<? extends DataBuffer> body) {
                            if (body instanceof Mono<? extends DataBuffer> fluxBody) {
                                return super.writeWith(fluxBody.map(buffer -> {
                                    // Log rs body
                                    var bytes = new byte[buffer.readableByteCount()];
                                    buffer.read(bytes);
                                    DataBufferUtils.release(buffer); // Release the buffer after reading
                                    var responseBody = new String(bytes, StandardCharsets.UTF_8);
                                    log.info("!!! Response Body: {}", responseBody);
                                    return rs.bufferFactory().wrap(bytes);
                                }));
                            } else if (body instanceof Flux<? extends DataBuffer> fluxBody) {
                                return super.writeWith(fluxBody.map(buffer -> {
                                    // Log rs body
                                    var bytes = new byte[buffer.readableByteCount()];
                                    buffer.read(bytes);
                                    DataBufferUtils.release(buffer); // Release the buffer after reading
                                    var responseBody = new String(bytes, StandardCharsets.UTF_8);
                                    log.info("!!! Response Body: {}", responseBody);
                                    return rs.bufferFactory().wrap(bytes);
                                }));
                            }
                            return super.writeWith(body);
                        }
                    };
                    return chain.filter(exchange.mutate()
                        .request(decoratedRq)
                        .response(decoratedRs)
                        .build());
                });
        };
    }
}