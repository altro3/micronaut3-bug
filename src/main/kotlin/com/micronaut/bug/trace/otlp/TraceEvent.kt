package com.micronaut.bug.trace.otlp

import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status

data class TraceEvent(
    val traceIdHex: String,
    val spanIdHex: String,
    val parentIdHex: String?,
    val name: String,
    val startEpochNanos: Long,
    val endEpochNanos: Long,
    val status: Status.StatusCode,
    val kind: Span.SpanKind,
    val userAttrs: Map<String, Any>?,
    val baggage: Map<String, String>?,
    val propagationHeaders: Map<String, String>?,
    val error: Throwable?
)