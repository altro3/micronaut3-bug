package com.micronaut.bug.trace.otlp

import com.google.protobuf.UnsafeByteOperations.unsafeWrap
import com.micronaut.bug.log.LogUtil.formatStackTrace
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_PROPAGATION
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
import java.util.concurrent.ConcurrentLinkedQueue

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

    private val contextPool = ConcurrentLinkedQueue<EncodingContext>()

    fun encodeBatch(events: List<TraceEvent>): ByteArray {
        val size = events.size
        if (size == 0) return EMPTY_BYTE_ARRAY

        val context = contextPool.poll() ?: EncodingContext()

        try {
            context.scopeSpansBuilder.clear()
                .setScope(libScope)

            for (i in 0 until size) {
                val event = events[i]

                val currentSpanBuilder = context.scopeSpansBuilder.addSpansBuilder()

                currentSpanBuilder.clear()
                    .setTraceId(unsafeWrap(event.traceIdBytes))
                    .setSpanId(unsafeWrap(event.spanIdBytes))
                    .setName(event.name)
                    .setKind(event.kind)
                    .setStartTimeUnixNano(event.startEpochNanos)
                    .setEndTimeUnixNano(event.endEpochNanos)
                    .setStatus(
                        context.statusBuilder.clear()
                            .setCode(event.status)
                    )

                if (event.hasParent) {
                    currentSpanBuilder.parentSpanId = unsafeWrap(event.parentIdBytes)
                }

                // Передаем точечный билдер для наполнения мап атрибутов
                context.addMapAttributes(currentSpanBuilder, event.userAttrs)
                context.addMapAttributes(currentSpanBuilder, event.baggage, PREFIX_BAGGAGE, context.baggageKeyCache)
                context.addMapAttributes(currentSpanBuilder, event.propagationHeaders, PREFIX_PROPAGATION, context.propKeyCache)

                event.error?.let {
                    context.stringBuilder.setLength(0)

                    currentSpanBuilder.addEvents(
                        context.eventBuilder.clear()
                            .setName(EVENT_NAME_EXCEPTION)
                            .setTimeUnixNano(event.endEpochNanos)
                            .addAttributes(context.createKeyValue(ATTR_EXCEPTION_TYPE, it.javaClass.name))
                            .addAttributes(context.createKeyValue(ATTR_EXCEPTION_MESSAGE, it.message ?: it.javaClass.simpleName))
                            .addAttributes(context.createKeyValue(ATTR_EXCEPTION_STACKTRACE, formatStackTrace(it, context.stringBuilder, props.stackTraceMaxLines, props.stackTraceRootCauseFull)))
                    )
                }
            }

            return context.requestBuilder.clear()
                .addResourceSpans(
                    context.resSpansBuilder.clear()
                        .setResource(serviceResource)
                        .addScopeSpans(context.scopeSpansBuilder.build())
                        .build()
                )
                .build()
                .toByteArray()
        } finally {
            contextPool.offer(context)
        }
    }

    class EncodingContext {
        val requestBuilder = ExportTraceServiceRequest.newBuilder()
        val resSpansBuilder = ResourceSpans.newBuilder()
        val scopeSpansBuilder = ScopeSpans.newBuilder()
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
                targetSpanBuilder.addAttributes(createKeyValue(targetKey, v))
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
    }

    companion object {
        const val STRING_NULL = "null"
        private val EMPTY_BYTE_ARRAY = ByteArray(0)
        const val EVENT_NAME_EXCEPTION = "exception"
        private const val MAX_CACHE_SIZE = 64
    }
}
