package com.micronaut.bug.trace.otlp

import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode

class TraceEvent {
    val traceIdBytes = ByteArray(16)
    val spanIdBytes = ByteArray(8)
    val parentIdBytes = ByteArray(8)

    var hasParent: Boolean = false
    var name: String = ""
    var startEpochNanos: Long = 0L
    var endEpochNanos: Long = 0L
    var status: StatusCode = StatusCode.STATUS_CODE_UNSET
    var kind: SpanKind = SpanKind.SPAN_KIND_INTERNAL

    var userAttrs: Map<String, Any>? = null
    var baggage: Map<String, String>? = null
    var propagationHeaders: Map<String, String>? = null
    var error: Throwable? = null

    fun update(
        traceIdHex: String, spanIdHex: String, parentIdHex: String?, name: String,
        startEpochNanos: Long, endEpochNanos: Long, status: StatusCode, kind: SpanKind,
        userAttrs: Map<String, Any>?, baggage: Map<String, String>?, propagationHeaders: Map<String, String>?,
        error: Throwable?,
    ) {
        parseHex(traceIdHex, traceIdBytes)
        parseHex(spanIdHex, spanIdBytes)

        if (!parentIdHex.isNullOrBlank()) {
            parseHex(parentIdHex, parentIdBytes)
            this.hasParent = true
        } else {
            this.hasParent = false
        }

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

    fun clearReferences() {
        this.name = ""
        this.hasParent = false
        this.userAttrs = null
        this.baggage = null
        this.propagationHeaders = null
        this.error = null
        traceIdBytes.fill(0)
        spanIdBytes.fill(0)
        parentIdBytes.fill(0)
    }

    private fun parseHex(hex: String, target: ByteArray) {
        val len = hex.length
        val bytesCount = minOf(len / 2, target.size)
        for (i in 0 until bytesCount) {
            val h = DECODE_TABLE[hex[i * 2].code and 0x7F]
            val l = DECODE_TABLE[hex[i * 2 + 1].code and 0x7F]
            target[i] = ((h shl 4) or l).toByte()
        }
    }

    companion object {
        private val DECODE_TABLE = IntArray(128).apply {
            for (i in 0..9) this['0'.code + i] = i
            for (i in 0..5) {
                this['a'.code + i] = 10 + i
                this['A'.code + i] = 10 + i
            }
        }
    }
}
