package com.micronaut.bug.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.Part;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpRequestDecorator;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE) // Должен быть первым, чтобы засечь время и поставить TraceId
public class LoggingFilter implements WebFilter {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {

        MediaType contentType = exchange.getRequest().getHeaders().getContentType();

        // Если это не multipart, используем обычную логику (как делали раньше)
        if (contentType == null || !contentType.includes(MediaType.MULTIPART_FORM_DATA)) {
            return processSimpleBody(exchange, chain);
        }
        return exchange.getMultipartData()
            .flatMap(map -> {
                List<Mono<String>> partLogMonos = new ArrayList<>();

                map.forEach((name, parts) -> {
                    for (Part part : parts) {
                        if (isExplicitBinary(part)) {
                            // Случай А: Явный файл или бинарный тип
                            String fileName = part.headers().getContentDisposition().getFilename();
                            String info = (fileName != null) ? "File: " + fileName : "Binary data";
                            partLogMonos.add(Mono.just("--- Part: " + name + " | " + info));
                        } else {
                            // Случай Б: Подозрение на текст (как ваш JSON из Postman без заголовков)
                            partLogMonos.add(extractTextOrBinaryAsync(name, part));
                        }
                    }
                });

                return Flux.concat(partLogMonos)
                    .collectList()
                    .doOnNext(logs -> log.info("PRE-LOG MULTIPART: {} {}\n{}",
                        exchange.getRequest().getMethod(), exchange.getRequest().getURI().getPath(), String.join("\n", logs)))
                    .then(chain.filter(exchange));
            });
    }

    private boolean isExplicitBinary(Part part) {
        // 1. Если есть имя файла — это 100% файл (Binary)
        if (part.headers().getContentDisposition().getFilename() != null) {
            return true;
        }
        // 2. Если есть Content-Type и он НЕ текстовый
        MediaType type = part.headers().getContentType();
        if (type != null) {
            return !type.includes(MediaType.APPLICATION_JSON) &&
                !type.includes(MediaType.TEXT_PLAIN) &&
                !type.getType().equals("text");
        }
        // В остальных случаях (нет заголовков вообще) — идем на проверку содержимого
        return false;
    }

    private Mono<String> extractTextOrBinaryAsync(String name, Part part) {
        return DataBufferUtils.join(part.content())
            .map(db -> {
                try {
                    int length = db.readableByteCount();
                    byte[] bytes = new byte[length];
                    // Читаем через ReadOnlyBuffer, чтобы не "сломать" позицию для контроллера
                    db.asByteBuffer().asReadOnlyBuffer().get(bytes);

                    // Эвристика: если нет нулевых байтов, считаем текстом
                    if (isTextContent(bytes)) {
                        String body = new String(bytes, StandardCharsets.UTF_8);
                        return "--- Part: " + name + " | Content: " + body;
                    } else {
                        return "--- Part: " + name + " | Binary data (No metadata)";
                    }
                } finally {
                    DataBufferUtils.release(db);
                }
            })
            .defaultIfEmpty("--- Part: " + name + " | Empty content");
    }

    private boolean isTextContent(byte[] bytes) {
        if (bytes.length == 0) {
            return true;
        }
        int checkLen = Math.min(bytes.length, 100);
        for (int i = 0; i < checkLen; i++) {
            if (bytes[i] == 0) {
                return false; // Признак бинарного файла
            }
        }
        return true;
    }

    private Mono<Void> processSimpleBody(ServerWebExchange exchange, WebFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getURI().getPath();
        String method = request.getMethod().name();

        return DataBufferUtils.join(exchange.getRequest().getBody())
            .flatMap(dataBuffer -> {
                byte[] bytes = new byte[dataBuffer.readableByteCount()];
                dataBuffer.asByteBuffer().asReadOnlyBuffer().get(bytes);
                DataBufferUtils.release(dataBuffer);

                String body = new String(bytes, StandardCharsets.UTF_8);
                log.info("PRE-LOG: {} {} | Body: {}", method, path, body);

                // КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
                // Вместо сложного декоратора используем простую мутацию тела
                ServerHttpRequest mutatedRequest = exchange.getRequest().mutate()
                    .header(HttpHeaders.CONTENT_LENGTH, String.valueOf(bytes.length))
                    .build();

                // Создаем новый exchange, подменяя тело через Flux.just
                ServerWebExchange mutatedExchange = exchange.mutate()
                    .request(new ServerHttpRequestDecorator(mutatedRequest) {
                        @Override
                        public Flux<DataBuffer> getBody() {
                            return Flux.just(DefaultDataBufferFactory.sharedInstance.wrap(bytes));
                        }
                    })
                    .build();

                return chain.filter(mutatedExchange);
            })
            .switchIfEmpty(Mono.defer(() -> chain.filter(exchange)))
            // Защита от дублирования ответа
            .onErrorResume(_ -> {
                if (!exchange.getResponse().isCommitted()) {
                    return chain.filter(exchange);
                }
                return Mono.empty();
            });
    }
}