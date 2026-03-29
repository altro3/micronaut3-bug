package com.micronaut.bug.config;

import lombok.extern.slf4j.Slf4j;
import org.reactivestreams.Publisher;
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
import java.util.List;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE) // Должен быть первым, чтобы засечь время и поставить TraceId
public class LoggingFilter implements WebFilter {

    private static final String EMPTY_BODY = "[Empty body]";

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {

        // Декорируем ответ
        var responseDecorator = new LoggingResponseDecorator(exchange.getResponse());
        MediaType contentType = exchange.getRequest().getHeaders().getContentType();
        var request = exchange.getRequest();
        var method = request.getMethod().name();
        var uri = request.getURI();
        var headers = request.getHeaders();

        // Если это не multipart, используем обычную логику (как делали раньше)
        if (contentType == null || !contentType.includes(MediaType.MULTIPART_FORM_DATA)) {
            return processSimpleBody(exchange, chain)
                .doFinally(signalType -> {
                    // Логируем, когда всё завершено (успех или ошибка)
                    logResponse(exchange, responseDecorator);
                });
        }
        return exchange.getMultipartData()
            .flatMap(map -> {
                List<Mono<String>> partLogMonos = new ArrayList<>();

                map.forEach((name, parts) -> {
                    for (Part part : parts) {
                        if (isExplicitBinary(part)) {
                            // Случай А: Явный файл или бинарный тип
                            String fileName = part.headers().getContentDisposition().getFilename();
                            String info = fileName != null ? "File: " + fileName : "Binary data";
                            partLogMonos.add(Mono.just("--- Part: " + name + " | " + info));
                        } else {
                            // Случай Б: Подозрение на текст (как ваш JSON из Postman без заголовков)
                            partLogMonos.add(extractTextOrBinaryAsync(name, part));
                        }
                    }
                });
                // Собираем все части в один структурированный лог
                return Flux.concat(partLogMonos)
                    .collectList()
                    .doOnNext(partLogs -> {
                        var fullBodyLog = String.join("\n", partLogs);

                        // Текстовый блок для удобного чтения лога
                        log.info("""
                                
                                ------------------ Service request ------------------
                                URI: {} {}
                                Headers: {}
                                Body:
                                {}
                                ------------------ /Service request ------------------
                                """,
                            method,
                            uri,
                            headers,
                            fullBodyLog
                        );
                    })
                    .then(chain.filter(exchange));
            })
            .doFinally(signalType -> {
                // Логируем, когда всё завершено (успех или ошибка)
                logResponse(exchange, responseDecorator);
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
                    var length = db.readableByteCount();
                    var bytes = new byte[length];
                    var offset = new int[] {0}; // Effectively final для использования в лямбде

                    // Java 25: Обработка сегментов через итератор без сдвига позиции чтения
                    db.readableByteBuffers().forEachRemaining(buffer -> {
                        var remaining = buffer.remaining();
                        // Читаем через Read-Only срез для безопасности Netty
                        buffer.asReadOnlyBuffer().get(bytes, offset[0], remaining);
                        offset[0] += remaining;
                    });

                    // Эвристика: если нет нулевых байтов, считаем текстом
                    if (isTextContent(bytes)) {
                        var body = new String(bytes, StandardCharsets.UTF_8);
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
        var request = exchange.getRequest();
        var uri = request.getURI();
        var method = request.getMethod().name();
        var headers = request.getHeaders();

        return DataBufferUtils.join(request.getBody())
            .flatMap(dataBuffer -> {
                var length = dataBuffer.readableByteCount();
                var bytes = new byte[length];
                // Используем AtomicInteger или массив для смещения, так как переменная в лямбде должна быть effectively final
                int[] offset = {0};

                // Java 25: эффективный перебор сегментов памяти без deprecated методов
                dataBuffer.readableByteBuffers().forEachRemaining(byteBuffer -> {
                    var remaining = byteBuffer.remaining();
                    byteBuffer.asReadOnlyBuffer().get(bytes, offset[0], remaining);
                    offset[0] += remaining;
                });


                // ВАЖНО: Освобождаем объединенный буфер после копирования
                DataBufferUtils.release(dataBuffer);

                String body = new String(bytes, StandardCharsets.UTF_8);
                log.info("""
                        
                        ------------------ Service request ------------------
                        URI: {} {}
                        Headers: {}
                        Body: {}
                        ------------------ /Service request ------------------
                        """,
                    method, uri,
                    headers,
                    body.isBlank() ? EMPTY_BODY : body
                );

                // КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
                // Вместо сложного декоратора используем простую мутацию тела
                ServerHttpRequest mutatedRequest = request.mutate()
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
            .switchIfEmpty(Mono.defer(() -> {
                // Лог для GET-запросов или POST без тела
                log.info("""
                    
                    ------------------ Service request ------------------
                    URI: {} {}
                    Headers: {}
                    Body: [Empty body]
                    ------------------ /Service request ------------------
                    """, method, uri, headers);
                return chain.filter(exchange);
            }))
            // Защита от дублирования ответа
            .onErrorResume(t -> {
                if (!exchange.getResponse().isCommitted()) {
                    return chain.filter(exchange);
                }
                return Mono.empty();
            });
    }

    private void logResponse(ServerWebExchange exchange, LoggingResponseDecorator decorator) {
        int status = exchange.getResponse().getStatusCode().value();
        var uri = exchange.getRequest().getURI();

        String body = decorator.getCachedBody();
        if (body.isEmpty() && isBinaryResponse(decorator)) {
            body = "[Binary data / Stream]";
        } else if (body.isEmpty()) {
            body = "[Empty body]";
        }

        log.info("OUT-LOG: {} | Status: {} | Body: {}", uri, status, body);
    }

    private boolean isBinaryResponse(LoggingResponseDecorator decorator) {
        MediaType type = decorator.getHeaders().getContentType();
        if (type == null) {
            return true;
        }
        return !type.includes(MediaType.APPLICATION_JSON) && !type.includes(MediaType.TEXT_PLAIN);
    }

    /**
     * Декоратор для перехвата тела ответа в Netty
     */
    class LoggingResponseDecorator extends ServerHttpResponseDecorator {

        private final StringBuilder bodyCollector = new StringBuilder();
        private static final int LIMIT = 8192000; // 8MB

        public LoggingResponseDecorator(ServerHttpResponse delegate) {
            super(delegate);
        }

        @Override
        public Mono<Void> writeWith(Publisher<? extends DataBuffer> body) {
            var contentType = getHeaders().getContentType();
            var isLoggable = contentType != null &&
                (contentType.includes(MediaType.APPLICATION_JSON) || contentType.includes(MediaType.TEXT_PLAIN));

            return super.writeWith(Flux.from(body).doOnNext(buffer -> {
                if (isLoggable && bodyCollector.length() < LIMIT) {
                    // Java 25: Используем прямое чтение через readableByteBuffers (Project Panama оптимизация)
                    buffer.readableByteBuffers().forEachRemaining(byteBuffer -> {
                        var readOnly = byteBuffer.asReadOnlyBuffer();
                        var bytes = new byte[readOnly.remaining()];
                        readOnly.get(bytes);
                        bodyCollector.append(new String(bytes, StandardCharsets.UTF_8));
                    });
                }
            }));
        }

        @Override
        public Mono<Void> writeAndFlushWith(Publisher<? extends Publisher<? extends DataBuffer>> body) {
            return writeWith(Flux.from(body).flatMap(Flux::from));
        }

        public String getCachedBody() {
            return bodyCollector.toString();
        }
    }
}