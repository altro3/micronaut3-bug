package com.micronaut.bug.trace.otlp

import com.micronaut.bug.log.LogUtil.formatStackTrace
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_PROPAGATION
import com.micronaut.bug.trace.TraceIdGenerator
import com.micronaut.bug.trace.config.TraceProperties
import io.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.InstrumentationScope
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.resource.v1.Resource
import io.opentelemetry.proto.trace.v1.ResourceSpans
import io.opentelemetry.proto.trace.v1.ScopeSpans
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status

class OtlpTraceEncoder(
    appName: String,
    nodeName: String,
    private val props: TraceProperties,
) {

    private val serviceResource = Resource.newBuilder()
        .addAttributes(KeyValue.newBuilder().setKey(NanoTracer.ATTR_SERVICE_NAME).setValue(AnyValue.newBuilder().setStringValue(appName).build()).build())
        .addAttributes(KeyValue.newBuilder().setKey(NanoTracer.ATTR_DEPLOYMENT_ENVIRONMENT).setValue(AnyValue.newBuilder().setStringValue(nodeName).build()).build())
        .build()
    private val libScope = InstrumentationScope.newBuilder()
        .setName("NanoTracer")
        .setVersion("1.0")
        .build()

    private val propKeyCache = object : LinkedHashMap<String, String>(MAX_CACHE_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: Map.Entry<String, String>?): Boolean = size > MAX_CACHE_SIZE
    }
    private val baggageKeyCache = object : LinkedHashMap<String, String>(MAX_CACHE_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: Map.Entry<String, String>?): Boolean = size > MAX_CACHE_SIZE
    }

    private val requestBuilder = ExportTraceServiceRequest.newBuilder()
    private val resSpansBuilder = ResourceSpans.newBuilder()
    private val scopeSpansBuilder = ScopeSpans.newBuilder()
    private val spanBuilder = Span.newBuilder()
    private val eventBuilder = Span.Event.newBuilder()
    private val statusBuilder = Status.newBuilder()
    private val valueBuilder = AnyValue.newBuilder()
    private val keyValueBuilder = KeyValue.newBuilder()
    private val stringBuilder = StringBuilder(2048)

    fun encodeBatch(events: List<TraceEvent>): ByteArray {
        val size = events.size
        if (size == 0) {
            return EMPTY_BYTE_ARRAY
        }

        scopeSpansBuilder.clear()
            .setScope(libScope)

        // ОПТИМИЗАЦИЯ: Чистый цикл без создания итераторов
        for (i in 0 until size) {
            val event = events[i]

            spanBuilder.clear()
                .setTraceId(TraceIdGenerator.toByteString(event.traceIdHex))
                .setSpanId(TraceIdGenerator.toByteString(event.spanIdHex))
                .setName(event.name)
                .setKind(event.kind)
                .setStartTimeUnixNano(event.startEpochNanos)
                .setEndTimeUnixNano(event.endEpochNanos)
                .setStatus(statusBuilder.clear().setCode(event.status))

            val parentId = event.parentIdHex
            if (!parentId.isNullOrBlank()) {
                spanBuilder.parentSpanId = TraceIdGenerator.toByteString(parentId)
            }

            // 1. Пользовательские атрибуты (без префикса и без кэша)
            spanBuilder.addMapAttributes(event.userAttrs)
            // 2. Динамический Багаж
            spanBuilder.addMapAttributes(event.baggage, PREFIX_BAGGAGE, baggageKeyCache)
            // 3. Заголовки пропогейшена
            spanBuilder.addMapAttributes(event.propagationHeaders, PREFIX_PROPAGATION, propKeyCache)

            event.error?.let {
                spanBuilder.addEvents(
                    eventBuilder.clear()
                        .setName(EVENT_NAME_EXCEPTION)
                        .setTimeUnixNano(event.endEpochNanos)
                        .addAttributes(createKeyValue(ATTR_EXCEPTION_TYPE, it.javaClass.name))
                        .addAttributes(createKeyValue(ATTR_EXCEPTION_MESSAGE, it.message ?: it.javaClass.simpleName))
                        .addAttributes(createKeyValue(ATTR_EXCEPTION_STACKTRACE, formatStackTrace(it, stringBuilder, props.stackTraceMaxLines, props.stackTraceRootCauseFull)))
                )
            }

            scopeSpansBuilder.addSpans(spanBuilder.build())
        }

        return requestBuilder.clear()
            .addResourceSpans(
                resSpansBuilder.clear()
                    .setResource(serviceResource)
                    .addScopeSpans(scopeSpansBuilder.build())
                    .build()
            )
            .build()
            .toByteArray()
    }

    private fun Span.Builder.addMapAttributes(
        sourceMap: Map<String, Any?>?,
        prefix: String = STRING_EMPTY,
        cache: MutableMap<String, String>? = null
    ) {
        if (sourceMap.isNullOrEmpty()) return

        for ((k, v) in sourceMap) {
            // Если есть префикс и кэш — берем из кэша, иначе используем ключ как есть
            val targetKey = if (prefix.isNotEmpty() && cache != null) {
                cache.getOrPut(k) {
                    if (k.startsWith(prefix)) k else "$prefix$k"
                }
            } else {
                k
            }
            this.addAttributes(createKeyValue(targetKey, v))
        }
    }

    private fun createKeyValue(key: String, value: Any?): KeyValue {
        valueBuilder.clear()
        buildAnyValue(value)
        return keyValueBuilder.clear()
            .setKey(key)
            .setValue(valueBuilder)
            .build()
    }

    private fun buildAnyValue(v: Any?) {
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

        const val STRING_EMPTY = ""
        const val STRING_NULL = "null"
        private val EMPTY_BYTE_ARRAY = ByteArray(0)
        const val EVENT_NAME_EXCEPTION = "exception"
        private const val MAX_CACHE_SIZE = 1000
    }
}
