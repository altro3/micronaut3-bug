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

    fun encodeBatch(events: List<TraceEvent>): ByteArray {
        val size = events.size
        if (size == 0) {
            return EMPTY_BYTE_ARRAY
        }

        val scopeSpansBuilder = tlScopeSpansBuilder.get()
        val spanBuilder = tlSpanBuilder.get()
        val statusBuilder = tlStatusBuilder.get()

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

            if (!event.parentIdHex.isNullOrBlank()) {
                spanBuilder.parentSpanId = TraceIdGenerator.toByteString(event.parentIdHex)
            }

            // Наполняем атрибуты
            event.userAttrs?.forEach { (k, v) ->
                spanBuilder.addAttributes(keyValue(k, v))
            }

            // Наполняем багаж
            event.baggage?.forEach { (k, v) ->
                spanBuilder.addAttributes(keyValue("$PREFIX_BAGGAGE$k", v))
            }

            event.propagationHeaders?.forEach { (k, v) ->
                spanBuilder.addAttributes(keyValue("$PREFIX_PROPAGATION$k", v))
            }

            event.error?.let {
                spanBuilder.addEvents(
                    tlEventBuilder.get().clear()
                        .setName(EVENT_NAME_EXCEPTION)
                        .setTimeUnixNano(event.endEpochNanos)
                        .addAttributes(keyValue(ATTR_EXCEPTION_TYPE, it.javaClass.name))
                        .addAttributes(keyValue(ATTR_EXCEPTION_MESSAGE, it.message ?: it.javaClass.simpleName))
                        .addAttributes(keyValue(ATTR_EXCEPTION_STACKTRACE, formatStackTrace(it, tlStringBuilder.get(), props.stackTraceMaxLines, props.stackTraceRootCauseFull)))
                )
            }

            scopeSpansBuilder.addSpans(spanBuilder.build())
        }

        return tlRequestBuilder.get().clear()
            .addResourceSpans(
                tlResSpansBuilder.get()
                    .clear()
                    .setResource(serviceResource)
                    .addScopeSpans(scopeSpansBuilder.build())
                    .build()
            )
            .build()
            .toByteArray()
    }

    private fun keyValue(k: String, v: Any): KeyValue {
        val vBuilder = tlValueBuilder.get().clear()

        when (v) {
            is String -> vBuilder.setStringValue(v)
            is Long -> vBuilder.setIntValue(v)
            is Int -> vBuilder.setIntValue(v.toLong())
            is Boolean -> vBuilder.setBoolValue(v)
            is Double -> vBuilder.setDoubleValue(v)
            is Float -> vBuilder.setDoubleValue(v.toDouble())
            is Iterable<*> -> {
                val arrayBuilder = vBuilder.arrayValueBuilder
                val itemValBuilder = tlItemValueBuilder.get()
                for (item in v) {
                    if (item != null) {
                        arrayBuilder.addValues(
                            itemValBuilder.clear()
                                .setStringValue(item.toString())
                                .build()
                        )
                    }
                }
            }

            else -> vBuilder.setStringValue(v.toString())
        }
        return tlKeyValueBuilder.get().clear()
            .setKey(k)
            .setValue(vBuilder)
            .build()
    }

    companion object {

        private val tlRequestBuilder = ThreadLocal.withInitial { ExportTraceServiceRequest.newBuilder() }
        private val tlResSpansBuilder = ThreadLocal.withInitial { ResourceSpans.newBuilder() }
        private val tlScopeSpansBuilder = ThreadLocal.withInitial { ScopeSpans.newBuilder() }
        private val tlSpanBuilder = ThreadLocal.withInitial { Span.newBuilder() }
        private val tlEventBuilder = ThreadLocal.withInitial { Span.Event.newBuilder() }
        private val tlStatusBuilder = ThreadLocal.withInitial { Status.newBuilder() }
        private val tlValueBuilder = ThreadLocal.withInitial { AnyValue.newBuilder() }
        private val tlKeyValueBuilder = ThreadLocal.withInitial { KeyValue.newBuilder() }
        private val tlItemValueBuilder = ThreadLocal.withInitial { AnyValue.newBuilder() }

        private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(2048) }
        private val EMPTY_BYTE_ARRAY = ByteArray(0)

        const val EVENT_NAME_EXCEPTION = "exception"
    }
}
