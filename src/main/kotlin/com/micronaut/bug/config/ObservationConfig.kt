package com.micronaut.bug.config

import io.micrometer.context.ContextRegistry
import org.slf4j.MDC
import org.springframework.context.annotation.Configuration

@Configuration
class ObservationConfig {

    init {
        val registry = ContextRegistry.getInstance()

        // Регистрируем кастомный ключ x-req-id
        registry.registerThreadLocalAccessor(
            X_REQ_ID,
            { MDC.get(X_REQ_ID) },
            { value -> MDC.put(X_REQ_ID, value) },
            { MDC.remove(X_REQ_ID) }
        )

        // Регистрируем кастомный ключ targetId
        registry.registerThreadLocalAccessor(
            TARGET_ID,
            { MDC.get(TARGET_ID) },
            { value -> MDC.put(TARGET_ID, value) },
            { MDC.remove(TARGET_ID) }
        )
    }

    companion object {
        const val X_REQ_ID = "x-req-id"
        const val TARGET_ID = "targetId"
    }
}
