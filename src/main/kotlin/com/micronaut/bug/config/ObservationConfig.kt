package com.micronaut.bug.config

import com.micronaut.bug.config.ServerLoggingFilter.Companion.MDC_RQ_ID
import com.micronaut.bug.config.log.MdcConverter.Companion.MDC_TARGET_ID
import io.micrometer.context.ContextRegistry
import org.slf4j.MDC

//@Configuration
class ObservationConfig {

    init {
        val registry = ContextRegistry.getInstance()

        // Регистрируем кастомный ключ x-req-id
        registry.registerThreadLocalAccessor(
            MDC_RQ_ID,
            { MDC.get(MDC_RQ_ID) },
            { value -> MDC.put(MDC_RQ_ID, value) },
            { MDC.remove(MDC_RQ_ID) }
        )

        // Регистрируем кастомный ключ targetId
        registry.registerThreadLocalAccessor(
            MDC_TARGET_ID,
            { MDC.get(MDC_TARGET_ID) },
            { value -> MDC.put(MDC_TARGET_ID, value) },
            { MDC.remove(MDC_TARGET_ID) }
        )
    }
}
