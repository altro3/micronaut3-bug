package com.micronaut.bug.trace

data class TraceContext(
    val traceId: String,
    val spanId: String,
    val parentId: String? = null,
    val name: String,
    val startEpochNanos: Long,
    val baggage: Map<String, String>? = null,
    val propagationHeaders: Map<String, String>? = null,
    val traceState: String? = null,
    @Volatile
    var error: Throwable? = null,
    val sampled: Boolean = true,
)
