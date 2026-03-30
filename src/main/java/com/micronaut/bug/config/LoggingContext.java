package com.micronaut.bug.config;

import org.slf4j.MDC;

public class LoggingContext {
    private static final ThreadLocal<String> TRACE_HOLDER = new ThreadLocal<>();
    public static final String X_REQ_ID = "X-Request-ID";

    public static void set(String traceId) {
        TRACE_HOLDER.set(traceId);
        // Дублируем в MDC для стандартных логгеров
        if (traceId != null) {
            MDC.put(X_REQ_ID, traceId);
        }
    }

    public static String get() {
        return TRACE_HOLDER.get();
    }

    public static void reset() {
        TRACE_HOLDER.remove();
        MDC.remove(X_REQ_ID);
    }
}