package com.micronaut.bug.trace.otlp

import com.google.protobuf.UnsafeByteOperations.unsafeWrap
import com.micronaut.bug.log.LogUtil.formatStackTrace
import com.micronaut.bug.trace.TraceUtil.ATTR_DEPLOYMENT_ENVIRONMENT
import com.micronaut.bug.trace.TraceUtil.ATTR_EXCEPTION_MESSAGE
import com.micronaut.bug.trace.TraceUtil.ATTR_EXCEPTION_STACKTRACE
import com.micronaut.bug.trace.TraceUtil.ATTR_EXCEPTION_TYPE
import com.micronaut.bug.trace.TraceUtil.ATTR_SERVICE_NAME
import com.micronaut.bug.trace.TraceUtil.PREFIX_BAGGAGE
import com.micronaut.bug.trace.TraceUtil.PREFIX_PROPAGATION
import com.micronaut.bug.trace.config.TraceProperties
import io.opentelemetry.proto.common.v1.AnyValue
import io.opentelemetry.proto.common.v1.InstrumentationScope
import io.opentelemetry.proto.common.v1.KeyValue
import io.opentelemetry.proto.resource.v1.Resource
import java.util.concurrent.ConcurrentLinkedQueue

class OtlpTraceEncoder(
    appName: String,
    nodeName: String,
    private val props: TraceProperties,
) {

    private val serviceResource = Resource.newBuilder()
        .addAttributes(KeyValue.newBuilder().setKey(ATTR_SERVICE_NAME).setValue(AnyValue.newBuilder().setStringValue(appName).build()).build())
        .addAttributes(KeyValue.newBuilder().setKey(ATTR_DEPLOYMENT_ENVIRONMENT).setValue(AnyValue.newBuilder().setStringValue(nodeName).build()).build())
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

                val currentSpanBuilder = context.spanBuilder.clear()

                currentSpanBuilder
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

                context.addMapAttributes(currentSpanBuilder, event.userAttrs)
                context.addMapAttributes(currentSpanBuilder, event.baggage, PREFIX_BAGGAGE, context.baggageKeyCache)
                context.addMapAttributes(currentSpanBuilder, event.propagationHeaders, PREFIX_PROPAGATION, context.propKeyCache)

                event.error?.let {
                    context.stringBuilder.setLength(0)

                    val exceptionEvent = context.eventBuilder.clear()
                        .setName(EVENT_NAME_EXCEPTION)
                        .setTimeUnixNano(event.endEpochNanos)
                        .addAttributes(context.createKeyValue(ATTR_EXCEPTION_TYPE, it.javaClass.name))
                        .addAttributes(context.createKeyValue(ATTR_EXCEPTION_MESSAGE, it.message ?: it.javaClass.simpleName))
                        .addAttributes(context.createKeyValue(ATTR_EXCEPTION_STACKTRACE, formatStackTrace(it, context.stringBuilder, props.stackTraceMaxLines, props.stackTraceRootCauseFull)))
                        .build()

                    currentSpanBuilder.addEvents(exceptionEvent)
                }

                context.scopeSpansBuilder.addSpans(currentSpanBuilder.build())
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

    companion object {
        const val STRING_NULL = "null"
        private val EMPTY_BYTE_ARRAY = ByteArray(0)
        const val EVENT_NAME_EXCEPTION = "exception"
    }
}
