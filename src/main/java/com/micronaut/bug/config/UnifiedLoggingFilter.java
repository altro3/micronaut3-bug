package com.micronaut.bug.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.reactivestreams.Publisher;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.HttpStatus;
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
import java.util.UUID;

import static com.micronaut.bug.config.ObservationConfig.X_REQ_ID;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
@RequiredArgsConstructor
public class UnifiedLoggingFilter implements WebFilter {

    private static final String BODY_EMPTY = "[Empty]";
    private static final String BODY_BINARY = "Binary data";
    private static final String BODY_NO_CONTENT = "No Body";
    private static final String STATUS_UNKNOWN = "UNKNOWN";
    private static final String PART_PREFIX = "  [PART] -> Name: %s | %s";
    private static final String PART_CONTENT = "Content: %s";
    private static final String PART_FILE = "File: %s";
    private static final String PART_EMPTY = "Empty content";
    private static final String MEDIA_TYPE_MAIN_TEXT = "text";

    private static final String SPACE = " ";
    private static final String NEW_LINE = "\n";
    private static final String MINUS = "-";
    private static final String EMPTY_STRING = "";

    private static final String LOG_TEMPLATE_RQ = """
        
        ------------------ Service request ------------------
        URI: {} {}
        Headers: {}
        Body:
        {}
        ------------------ /Service request ------------------
        """;

    private static final String LOG_TEMPLATE_RS = """
        
        ------------------ Service response ------------------
        URI: {}
        Status: {}
        Headers: {}
        Body: {}
        ------------------ /Service response ------------------
        """;

    private static final int LIMIT_LOG_SIZE = 8192;

    private final ObjectMapper objectMapper;
    private ObjectMapper prettyMapper;

    @PostConstruct
    void init() {
        this.prettyMapper = objectMapper.copy().enable(SerializationFeature.INDENT_OUTPUT);
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        var rq = exchange.getRequest();

        var requestId = rq.getHeaders().getFirst(X_REQ_ID);
        if (requestId == null || requestId.isBlank()) {
            requestId = UUID.randomUUID().toString().replace(MINUS, EMPTY_STRING);
        }
        final String finalId = requestId;

        return chain.filter(exchange)
            .doFinally(signal -> LoggingContext.reset()) // Очистка при завершении
            .contextWrite(ctx -> ctx.put(X_REQ_ID, finalId)); // Проброс в Reactor Context для Observation/Tracing

/*
        var rsDecorator = new LoggingRsDecorator(exchange.getResponse());
        var contentType = rq.getHeaders().getContentType();
        var isMultipart = contentType != null && contentType.includes(MediaType.MULTIPART_FORM_DATA);

        var exchangeMono = isMultipart ? processMultipart(exchange) : processSimple(exchange);

        return exchangeMono.flatMap(mutatedExchange ->
            chain.filter(mutatedExchange.mutate().response(rsDecorator).build())
                .doFinally(signal -> {
                    // Логируем результат
                    var code = exchange.getResponse().getStatusCode();
                    var statusInfo = (code instanceof HttpStatus hs)
                        ? hs.value() + SPACE + hs.getReasonPhrase()
                        : (code != null ? String.valueOf(code.value()) : STATUS_UNKNOWN);

                    var rsBody = formatIfJson(rsDecorator.getCachedBody(), rsDecorator.getHeaders().getContentType());

                    log.info(LOG_TEMPLATE_RS, rq.getURI(), statusInfo, rsDecorator.getHeaders(), rsBody.isBlank() ? BODY_EMPTY : rsBody);

                    // УДАЛЯЕМ ОДИН РАЗ В КОНЦЕ
                    LoggingContext.reset();
                })
        ).contextWrite(ctx -> ctx.put(X_REQ_ID, finalId)); // Проброс в реактивный контекст
*/
    }

    private Mono<ServerWebExchange> processSimple(ServerWebExchange exchange) {
        var rq = exchange.getRequest();
        return DataBufferUtils.join(rq.getBody())
            .flatMap(db -> {
                var bytes = readBytes(db);
                DataBufferUtils.release(db);
                var formattedBody = formatIfJson(new String(bytes, StandardCharsets.UTF_8), rq.getHeaders().getContentType());

                log.info(LOG_TEMPLATE_RQ, rq.getMethod(), rq.getURI(), rq.getHeaders(), formattedBody.isBlank() ? BODY_EMPTY : formattedBody);

                var mutatedRequest = new ServerHttpRequestDecorator(rq) {
                    @Override
                    public Flux<DataBuffer> getBody() {
                        return Flux.just(DefaultDataBufferFactory.sharedInstance.wrap(bytes));
                    }
                };
                return Mono.just(exchange.mutate().request(mutatedRequest).build());
            })
            .switchIfEmpty(Mono.defer(() -> {
                log.info(LOG_TEMPLATE_RQ, rq.getMethod(), rq.getURI(), rq.getHeaders(), BODY_NO_CONTENT);
                return Mono.just(exchange);
            }));
    }

    private Mono<ServerWebExchange> processMultipart(ServerWebExchange exchange) {
        var rq = exchange.getRequest();
        return exchange.getMultipartData().flatMap(map -> {
            var partLogMonos = new ArrayList<Mono<String>>();
            map.forEach((name, parts) -> {
                for (var part : parts) {
                    var fileName = part.headers().getContentDisposition().getFilename();
                    if (fileName != null) {
                        partLogMonos.add(Mono.just(PART_PREFIX.formatted(name, PART_FILE.formatted(fileName))));
                    } else {
                        partLogMonos.add(extractPartText(name, part));
                    }
                }
            });

            return Flux.concat(partLogMonos).collectList().map(logs -> {
                log.info(LOG_TEMPLATE_RQ, rq.getMethod(), rq.getURI(), rq.getHeaders(), String.join(NEW_LINE, logs));
                return exchange;
            });
        });
    }

    private Mono<String> extractPartText(String name, org.springframework.http.codec.multipart.Part part) {
        return DataBufferUtils.join(part.content()).map(db -> {
            try {
                var bytes = readBytes(db);
                if (isText(bytes)) {
                    var formatted = formatIfJson(new String(bytes, StandardCharsets.UTF_8), part.headers().getContentType());
                    return PART_PREFIX.formatted(name, PART_CONTENT.formatted(formatted));
                }
                return PART_PREFIX.formatted(name, BODY_BINARY);
            } finally {
                DataBufferUtils.release(db);
            }
        }).defaultIfEmpty(PART_PREFIX.formatted(name, PART_EMPTY));
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

    private static boolean isText(byte[] bytes) {
        if (bytes.length == 0) {
            return true;
        }
        for (int i = 0; i < Math.min(bytes.length, 100); i++) {
            if (bytes[i] == 0) {
                return false;
            }
        }
        return true;
    }

    private String formatIfJson(String body, MediaType type) {
        if (body.isBlank() || (type != null && !type.includes(MediaType.APPLICATION_JSON))) {
            return body;
        }
        try {
            return prettyMapper.writerWithDefaultPrettyPrinter().writeValueAsString(prettyMapper.readTree(body));
        } catch (Exception e) {
            return body;
        }
    }

    private static class LoggingRsDecorator extends ServerHttpResponseDecorator {

        private final StringBuilder body = new StringBuilder();

        public LoggingRsDecorator(ServerHttpResponse delegate) {
            super(delegate);
        }

        @Override
        public Mono<Void> writeWith(Publisher<? extends DataBuffer> body) {
            var type = getHeaders().getContentType();
            var loggable = type != null && (type.includes(MediaType.APPLICATION_JSON) || type.getType().equals(MEDIA_TYPE_MAIN_TEXT));
            return super.writeWith(Flux.from(body).doOnNext(db -> {
                if (loggable && this.body.length() < LIMIT_LOG_SIZE) {
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
