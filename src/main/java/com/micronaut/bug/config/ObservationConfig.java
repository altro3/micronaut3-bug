package com.micronaut.bug.config;

import io.micrometer.context.ContextRegistry;
import jakarta.annotation.PostConstruct;
import org.slf4j.MDC;
import org.springframework.context.annotation.Configuration;
//import reactor.core.publisher.Hooks;

@Configuration
public class ObservationConfig {

    public static final String X_REQ_ID = "x-req-id";
    public static final String TARGET_ID = "targetId";

    @PostConstruct
    public void init() {
        // 1. Включаем автоматический проброс в Reactor
        //        Hooks.enableAutomaticContextPropagation();

        var registry = ContextRegistry.getInstance();

        // 2. Регистрируем наш кастомный ключ x-req-id
        registry.registerThreadLocalAccessor(X_REQ_ID,
            () -> MDC.get(X_REQ_ID),
            val -> MDC.put(X_REQ_ID, val),
            () -> MDC.remove(X_REQ_ID)
        );

        registry.registerThreadLocalAccessor(TARGET_ID,
            () -> MDC.get(TARGET_ID),
            val -> MDC.put(TARGET_ID, val),
            () -> MDC.remove(TARGET_ID)
        );
    }
}