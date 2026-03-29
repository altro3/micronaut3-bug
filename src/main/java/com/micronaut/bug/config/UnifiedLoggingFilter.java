package com.micronaut.bug.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.reactivestreams.Publisher;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpRequestDecorator;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.http.server.reactive.ServerHttpResponseDecorator;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
@RequiredArgsConstructor
public class UnifiedLoggingFilter implements WebFilter {

    private final ObjectMapper objectMapper;
    private ObjectMapper prettyMapper;
    private static final String EMPTY_BODY = "[Empty]";
    private static final String PART_PREFIX = "  [PART] -> ";

    @PostConstruct
    void init() {
        this.prettyMapper = objectMapper.copy().enable(SerializationFeature.INDENT_OUTPUT);
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        var request = exchange.getRequest();
        var contentType = request.getHeaders().getContentType();
        var resDecorator = new LoggingResponseDecorator(exchange.getResponse());

        var isMultipart = contentType != null && contentType.includes(MediaType.MULTIPART_FORM_DATA);
        var exchangeMono = isMultipart ? processMultipart(exchange) : processSimple(exchange);

        return exchangeMono.flatMap(mutatedExchange ->
            chain.filter(mutatedExchange.mutate().response(resDecorator).build())
                .doFinally(signal -> {
                    var uri = request.getURI().toString();
                    var status = exchange.getResponse().getStatusCode().value();
                    var resBody = formatIfJson(resDecorator.getCachedBody(), resDecorator.getHeaders().getContentType());

                    log.info("""
                        
                        ------------------ Service response ------------------
                        URI: {}
                        Status: {}
                        Body: {}
                        ------------------ /Service response ------------------
                        """, uri, status, resBody.isBlank() ? EMPTY_BODY : resBody);
                })
        );
    }

    private Mono<ServerWebExchange> processMultipart(ServerWebExchange exchange) {
        var request = exchange.getRequest();
        return exchange.getMultipartData().flatMap(map -> {
            var partLogMonos = new ArrayList<Mono<String>>();
            map.forEach((name, parts) -> {
                for (var part : parts) {
                    if (isBinary(part)) {
                        var fileName = part.headers().getContentDisposition().getFilename();
                        var info = (fileName != null) ? "File: " + fileName : "Binary data";
                        partLogMonos.add(Mono.just("%sName: %s | %s".formatted(PART_PREFIX, name, info)));
                    } else {
                        partLogMonos.add(extractPartText(name, part));
                    }
                }
            });

            return Flux.concat(partLogMonos).collectList().map(logs -> {
                log.info("""
                    
                    ------------------ Service request ------------------
                    URI: {} {}
                    Headers: {}
                    Body (Multipart):
                    {}
                    ------------------ /Service request ------------------
                    """, request.getMethod(), request.getURI(), request.getHeaders(), String.join("\n", logs));
                return exchange;
            });
        });
    }

    private Mono<String> extractPartText(String name, org.springframework.http.codec.multipart.Part part) {
        return DataBufferUtils.join(part.content()).map(db -> {
            try {
                var raw = new String(readBytes(db), StandardCharsets.UTF_8);
                var formatted = formatIfJson(raw, part.headers().getContentType());
                return "%sName: %s | Content: %s".formatted(PART_PREFIX, name, formatted);
            } finally {
                DataBufferUtils.release(db);
            }
        });
    }

    private Mono<ServerWebExchange> processSimple(ServerWebExchange exchange) {
        var request = exchange.getRequest();
        return DataBufferUtils.join(request.getBody())
            .flatMap(db -> {
                var bytes = readBytes(db);
                DataBufferUtils.release(db);
                var rawBody = new String(bytes, StandardCharsets.UTF_8);
                var formattedBody = formatIfJson(rawBody, request.getHeaders().getContentType());

                log.info("""
                        
                        ------------------ Service request ------------------
                        URI: {} {}
                        Headers: {}
                        Body: {}
                        ------------------ /Service request ------------------
                        """, request.getMethod(), request.getURI(), request.getHeaders(),
                    formattedBody.isBlank() ? EMPTY_BODY : formattedBody);

                var mutatedRequest = new ServerHttpRequestDecorator(request) {
                    @Override
                    public Flux<DataBuffer> getBody() {
                        return Flux.just(DefaultDataBufferFactory.sharedInstance.wrap(bytes));
                    }
                };
                return Mono.just(exchange.mutate().request(mutatedRequest).build());
            })
            .switchIfEmpty(Mono.defer(() -> {
                log.info("""
                    
                    ------------------ Service request ------------------
                    URI: {} {}
                    Headers: {}
                    Body: {}
                    ------------------ /Service request ------------------
                    """, request.getMethod(), request.getURI(), request.getHeaders(), EMPTY_BODY);
                return Mono.just(exchange);
            }));
    }

    private String formatIfJson(String body, MediaType type) {
        if (body.isBlank() || type == null || !type.includes(MediaType.APPLICATION_JSON)) {
            return body;
        }
        try {
            return prettyMapper.writerWithDefaultPrettyPrinter().writeValueAsString(prettyMapper.readTree(body));
        } catch (Exception e) {
            return body;
        }
    }

    private static byte[] readBytes(DataBuffer db) {
        var bytes = new byte[db.readableByteCount()];
        var offset = new int[] {0};
        db.readableByteBuffers().forEachRemaining(buf -> {
            var readOnly = buf.asReadOnlyBuffer();
            var len = readOnly.remaining();
            readOnly.get(bytes, offset[0], len);
            offset[0] += len;
        });
        return bytes;
    }

    private static boolean isBinary(org.springframework.http.codec.multipart.Part part) {
        var fileName = part.headers().getContentDisposition().getFilename();
        var type = part.headers().getContentType();
        return fileName != null || (type != null && !type.includes(MediaType.APPLICATION_JSON) && !type.includes(MediaType.TEXT_PLAIN));
    }

    /**
     * Вложенный статический класс декоратора ответа
     */
    private static class LoggingResponseDecorator extends ServerHttpResponseDecorator {
        private final StringBuilder body = new StringBuilder();
        private static final int LIMIT = 8192;

        public LoggingResponseDecorator(ServerHttpResponse delegate) {
            super(delegate);
        }

        @Override
        public Mono<Void> writeWith(Publisher<? extends DataBuffer> body) {
            var type = getHeaders().getContentType();
            var loggable = type != null && (type.includes(MediaType.APPLICATION_JSON) || type.getType().equals("text"));
            return super.writeWith(Flux.from(body).doOnNext(db -> {
                if (loggable && this.body.length() < LIMIT) {
                    db.readableByteBuffers().forEachRemaining(buf -> {
                        var readOnly = buf.asReadOnlyBuffer();
                        var bytes = new byte[readOnly.remaining()];
                        readOnly.get(bytes);
                        this.body.append(new String(bytes, StandardCharsets.UTF_8));
                    });
                }
            }));
        }

        @Override
        public Mono<Void> writeAndFlushWith(Publisher<? extends Publisher<? extends DataBuffer>> body) {
            return writeWith(Flux.from(body).flatMap(Flux::from));
        }

        public String getCachedBody() {
            return body.toString();
        }
    }
}
