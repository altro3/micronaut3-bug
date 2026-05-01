package com.micronaut.bug.trace.otlp

import ch.qos.logback.classic.spi.ThrowableProxy
import ch.qos.logback.classic.spi.ThrowableProxyUtil
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.NanoTracer.Companion.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_BAGGAGE
import com.micronaut.bug.trace.NanoTracer.Companion.PREFIX_PROPAGATION
import com.micronaut.bug.trace.TraceIdGenerator
import io.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.ArrayValue
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
) {

    // Ресурс собирается один раз при старте
    private val serviceResource = Resource.newBuilder()
        .addAttributes(KeyValue.newBuilder().setKey(NanoTracer.ATTR_SERVICE_NAME).setValue(AnyValue.newBuilder().setStringValue(appName).build()).build())
        .addAttributes(KeyValue.newBuilder().setKey(NanoTracer.ATTR_DEPLOYMENT_ENVIRONMENT).setValue(AnyValue.newBuilder().setStringValue(nodeName).build()).build())
        .build()
    private val libScope = InstrumentationScope.newBuilder()
        .setName("NanoTracer")
        .setVersion("1.0.0")
        .build()

    /**
     * Преобразует батч легковесных событий в Protobuf-пакет в формате OTLP [5.3].
     */
    fun encodeBatch(events: List<TraceEvent>): ByteArray {
        val size = events.size
        if (size == 0) {
            return EMPTY_BYTE_ARRAY
        }

        val spanBuilder = tlSpanBuilder.get()
        val scopeSpansBuilder = ScopeSpans.newBuilder()
            .setScope(libScope)

        val sb = tlStringBuilder.get()
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
                .setStatus(Status.newBuilder().setCode(event.status))

            if (!event.parentIdHex.isNullOrBlank()) {
                spanBuilder.parentSpanId = TraceIdGenerator.toByteString(event.parentIdHex)
            }

            // Наполняем атрибуты
            if (!event.userAttrs.isNullOrEmpty()) {
                for (entry in event.userAttrs.entries) {
                    fillProtobufAttr(spanBuilder.addAttributesBuilder(), entry.key, entry.value)
                }
            }

            // Наполняем багаж
            if (!event.baggage.isNullOrEmpty()) {
                for (entry in event.baggage.entries) {
                    sb.setLength(0)
                    sb.append(PREFIX_BAGGAGE).append(entry.key)
                    val attr = spanBuilder.addAttributesBuilder()
                    attr.key = sb.toString()
                    attr.valueBuilder.stringValue = entry.value
                }
            }

            // Наполняем проброс заголовков
            if (!event.propagationHeaders.isNullOrEmpty()) {
                for (entry in event.propagationHeaders.entries) {
                    sb.setLength(0)
                    sb.append(PREFIX_PROPAGATION).append(entry.key)
                    val attr = spanBuilder.addAttributesBuilder()
                    attr.key = sb.toString()
                    attr.valueBuilder.stringValue = entry.value
                }
            }

            // Обрабатываем ошибку
            event.error?.let {

                val eventBuilder = spanBuilder.addEventsBuilder()
                // Стандарт требует называть это событие именно "exception"
                eventBuilder.name = EVENT_NAME_EXCEPTION
                // Время фиксации ошибки (в наносекундах)
                eventBuilder.timeUnixNano = event.endEpochNanos

                val typeAttr = spanBuilder.addAttributesBuilder()
                typeAttr.key = ATTR_EXCEPTION_TYPE
                typeAttr.valueBuilder.stringValue = it.javaClass.name

                val msgAttr = spanBuilder.addAttributesBuilder()
                msgAttr.key = ATTR_EXCEPTION_MESSAGE
                msgAttr.valueBuilder.stringValue = it.message ?: it.javaClass.simpleName

                val stackAttr = spanBuilder.addAttributesBuilder()
                stackAttr.key = ATTR_EXCEPTION_STACKTRACE
                stackAttr.valueBuilder.stringValue = ThrowableProxyUtil.asString(ThrowableProxy(it))
            }

            scopeSpansBuilder.addSpans(spanBuilder.build())
        }

        return ExportTraceServiceRequest.newBuilder()
            .addResourceSpans(
                ResourceSpans.newBuilder()
                    .setResource(serviceResource)
                    .addScopeSpans(scopeSpansBuilder.build())
                    .build()
            )
            .build()
            .toByteArray()
    }

    private fun fillProtobufAttr(attrBuilder: KeyValue.Builder, k: String, v: Any) {
        attrBuilder.key = k
        val valueBuilder = attrBuilder.valueBuilder
        when (v) {
            is String -> valueBuilder.stringValue = v
            is Long -> valueBuilder.intValue = v
            is Int -> valueBuilder.intValue = v.toLong()
            is Boolean -> valueBuilder.boolValue = v
            is Double -> valueBuilder.doubleValue = v
            is Float -> valueBuilder.doubleValue = v.toDouble()
            is Iterable<*> -> {
                val arrayBuilder = ArrayValue.newBuilder()
                v.forEach { item ->
                    if (item != null) {
                        arrayBuilder.addValues(AnyValue.newBuilder().setStringValue(item.toString()).build())
                    }
                }
                valueBuilder.setArrayValue(arrayBuilder)
            }

            else -> valueBuilder.stringValue = v.toString()
        }
    }

    companion object {
        // Билдер для переиспользования Protobuf структур
        private val tlSpanBuilder = ThreadLocal.withInitial { Span.newBuilder() }

        // Билдер для склейки строк без выделения мусора в Heap
        private val tlStringBuilder = ThreadLocal.withInitial { StringBuilder(64) }
        private val EMPTY_BYTE_ARRAY = ByteArray(0)

        // Название события для исключений по стандарту OpenTelemetry
        const val EVENT_NAME_EXCEPTION = "exception"
    }
}
