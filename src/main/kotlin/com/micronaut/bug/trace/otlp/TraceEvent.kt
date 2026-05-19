package com.micronaut.bug.trace.otlp

import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode

class TraceEvent(
    var traceIdHex: String = "",
    var spanIdHex: String = "",
    var parentIdHex: String? = null,
    var name: String = "",
    var startEpochNanos: Long = 0L,
    var endEpochNanos: Long = 0L,
    var status: StatusCode = StatusCode.STATUS_CODE_UNSET,
    var kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL,
    var userAttrs: Map<String, Any>? = null,
    var baggage: Map<String, String>? = null,
    var propagationHeaders: Map<String, String>? = null,
    var error: Throwable? = null
) {
    // Метод для переиспользования инстанса
    fun update(
        traceIdHex: String, spanIdHex: String, parentIdHex: String?, name: String,
        startEpochNanos: Long, endEpochNanos: Long, status: StatusCode, kind: SpanKind,
        userAttrs: Map<String, Any>?, baggage: Map<String, String>?, propagationHeaders: Map<String, String>?,
        error: Throwable?
    ) {
        this.traceIdHex = traceIdHex
        this.spanIdHex = spanIdHex
        this.parentIdHex = parentIdHex
        this.name = name
        this.startEpochNanos = startEpochNanos
        this.endEpochNanos = endEpochNanos
        this.status = status
        this.kind = kind
        this.userAttrs = userAttrs
        this.baggage = baggage
        this.propagationHeaders = propagationHeaders
        this.error = error
    }

    // Метод для очистки внешних тяжелых ссылок перед возвратом в пул
    fun clearReferences() {
        this.traceIdHex = ""
        this.spanIdHex = ""
        this.parentIdHex = null
        this.name = ""

        this.userAttrs = null
        this.baggage = null
        this.propagationHeaders = null
        this.error = null
    }
}
