package com.micronaut.bug.config.trace

/**
 * Контекст спана, хранящий идентификаторы и время начала.
 */
data class TraceContext(
    val traceId: String,
    val spanId: String,
    val parentId: String?,
    val name: String,
    val startEpochNanos: Long,
)
