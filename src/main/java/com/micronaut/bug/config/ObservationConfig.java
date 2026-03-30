package com.micronaut.bug.config;

import io.micrometer.context.ContextRegistry;
import jakarta.annotation.PostConstruct;
import org.slf4j.MDC;
import org.springframework.boot.web.embedded.netty.NettyServerCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.core.publisher.Hooks;
import reactor.netty.http.HttpProtocol;

@Configuration
public class ObservationConfig {

    public static final String X_REQ_ID = "x-req-id";
    public static final String TARGET_ID = "targetId";

    @PostConstruct
    public void init() {
        // 1. Включаем автоматический проброс в Reactor
        Hooks.enableAutomaticContextPropagation();

        var registry = ContextRegistry.getInstance();

        // 2. Регистрируем наш кастомный ключ x-req-id
        registry.registerThreadLocalAccessor(
            X_REQ_ID,
            LoggingContext::get,
            LoggingContext::set,
            LoggingContext::reset
        );

        registry.registerThreadLocalAccessor(TARGET_ID,
            () -> MDC.get(TARGET_ID),
            val -> MDC.put(TARGET_ID, val),
            () -> MDC.remove(TARGET_ID)
        );

        registry.registerThreadLocalAccessor(
            "current-user",
            UserContext::get,
            UserContext::set, // Используем временный ThreadLocal как транспорт
            UserContext::reset
        );
    }

//    @Bean
//    public NettyServerCustomizer http2CleartextCustomizer() {
//        return server -> server.protocol(HttpProtocol.H2C)
//            .wiretap(true);
//    }
}
