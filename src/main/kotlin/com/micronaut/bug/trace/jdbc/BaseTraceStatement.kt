package com.micronaut.bug.trace.jdbc

import com.micronaut.bug.trace.NanoSpan
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.TraceUtil
import com.micronaut.bug.trace.config.TraceProperties.TraceJdbcProperties
import com.micronaut.bug.trace.jdbc.JdbcTraceKeywords.parseOperation
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode

abstract class BaseTraceStatement(
    protected val tracer: NanoTracer,
    protected val traceProps: TraceJdbcProperties,
) {

    protected inline fun <T> executeWithTrace(sql: String, block: () -> T): T {
        val operation = parseOperation(sql)
        val span = tracer.startSpan(operation)

        try {
            val res = block()
            stopSpan(span, sql, operation, null)
            return res
        } catch (e: Throwable) {
            stopSpan(span, sql, operation, e)
            throw e
        }
    }

    protected fun stopSpan(span: NanoSpan, sql: String?, operation: String, error: Throwable?) {
        val isError = error != null
        if (!span.sampled && !isError) {
            tracer.stop(span, StatusCode.STATUS_CODE_OK, SpanKind.SPAN_KIND_CLIENT, null)
            return
        }

        val attrs = HashMap<String, Any>(4)
        attrs[TraceUtil.ATTR_DB_SYSTEM] = traceProps.system
        attrs[TraceUtil.ATTR_DB_OPERATION] = operation

        if (sql != null) {
            attrs[TraceUtil.ATTR_DB_STATEMENT] = if (sql.length > traceProps.maxStatementLength) {
                sql.substring(0, traceProps.maxStatementLength) + traceProps.truncatedMarker
            } else sql
        }

        if (isError) {
            span.error = error
        }

        tracer.stop(
            span = span,
            status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_OK,
            kind = SpanKind.SPAN_KIND_CLIENT,
            attrs = attrs
        )
    }
}
