package com.micronaut.bug.trace.otlp

import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_REQUEST_HEADER
import com.micronaut.bug.trace.TraceUtil.PREFIX_HTTP_RESPONSE_HEADER
import com.micronaut.bug.trace.otlp.OtlpTraceEncoder.Companion.STRING_NULL
import io.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.trace.v1.ResourceSpans
import io.opentelemetry.proto.trace.v1.ScopeSpans
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status

class EncodingContext {
    val requestBuilder = ExportTraceServiceRequest.newBuilder()
    val resSpansBuilder = ResourceSpans.newBuilder()
    val scopeSpansBuilder = ScopeSpans.newBuilder()

    val spanBuilder = Span.newBuilder()
    val eventBuilder = Span.Event.newBuilder()

    val statusBuilder = Status.newBuilder()
    val valueBuilder = AnyValue.newBuilder()
    val keyValueBuilder = KeyValue.newBuilder()
    val stringBuilder = StringBuilder(2048)

    val propKeyCache = object : LinkedHashMap<String, String>(MAX_CACHE_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: Map.Entry<String, String>?): Boolean = size > MAX_CACHE_SIZE
    }
    val baggageKeyCache = object : LinkedHashMap<String, String>(MAX_CACHE_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: Map.Entry<String, String>?): Boolean = size > MAX_CACHE_SIZE
    }

    fun addMapAttributes(
        targetSpanBuilder: Span.Builder,
        sourceMap: Map<String, Any?>?,
        prefix: String = "",
        cache: MutableMap<String, String>? = null
    ) {
        if (sourceMap.isNullOrEmpty()) return

        for ((k, v) in sourceMap) {
            val targetKey = if (prefix.isNotEmpty() && cache != null) {
                cache.getOrPut(k) {
                    if (k.startsWith(prefix)) k else "$prefix$k"
                }
            } else {
                k
            }

            val isHttpHeader = targetKey.startsWith(PREFIX_HTTP_REQUEST_HEADER) || targetKey.startsWith(PREFIX_HTTP_RESPONSE_HEADER)

            if (isHttpHeader && v is String) {
                targetSpanBuilder.addAttributes(createKeyValueForOtelHeader(targetKey, v))
            } else {
                targetSpanBuilder.addAttributes(createKeyValue(targetKey, v))
            }
        }
    }

    fun createKeyValue(key: String, value: Any?): KeyValue {
        valueBuilder.clear()
        buildAnyValue(value)
        return keyValueBuilder.clear()
            .setKey(key)
            .setValue(valueBuilder)
            .build()
    }

    private fun createKeyValueForOtelHeader(key: String, value: String): KeyValue {
        valueBuilder.clear()
        val arrayBuilder = valueBuilder.arrayValueBuilder
        arrayBuilder.addValuesBuilder().setStringValue(value)

        return keyValueBuilder.clear()
            .setKey(key)
            .setValue(valueBuilder)
            .build()
    }

    fun buildAnyValue(v: Any?) {
        if (v == null) {
            valueBuilder.setStringValue(STRING_NULL)
            return
        }
        when (v) {
            is String -> valueBuilder.setStringValue(v)
            is Long -> valueBuilder.setIntValue(v)
            is Int -> valueBuilder.setIntValue(v.toLong())
            is Boolean -> valueBuilder.setBoolValue(v)
            is Double -> valueBuilder.setDoubleValue(v)
            is Float -> valueBuilder.setDoubleValue(v.toDouble())
            is Iterable<*> -> {
                val arrayBuilder = valueBuilder.arrayValueBuilder
                for (item in v) {
                    if (item != null) {
                        arrayBuilder.addValuesBuilder().setStringValue(item.toString())
                    }
                }
            }

            else -> valueBuilder.setStringValue(v.toString())
        }
    }

    companion object {
        private const val MAX_CACHE_SIZE = 64
    }
}
