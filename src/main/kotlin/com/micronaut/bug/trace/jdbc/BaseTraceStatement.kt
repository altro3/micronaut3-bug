package com.micronaut.bug.trace.jdbc

import com.micronaut.bug.trace.NanoSpan
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.TraceUtil
import com.micronaut.bug.trace.config.TraceProperties.TraceJdbcProperties
import com.micronaut.bug.trace.jdbc.JdbcUrlParser.ConnectionInfo
import io.opentelemetry.proto.trace.v1.Span.SpanKind
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import java.sql.SQLException

abstract class BaseTraceStatement(
    protected val tracer: NanoTracer,
    protected val traceProps: TraceJdbcProperties,
    protected val connectionInfo: ConnectionInfo,
) {

    protected inline fun <T> executeWithTrace(sql: String, block: () -> T): T {
        val (operation, collectionName) = JdbcTraceKeywords.parseOperationAndCollection(sql)
        val spanName = if (collectionName != null) "$operation $collectionName" else operation
        val span = tracer.startSpan(spanName)

        try {
            val res = block()
            stopSpan(span, sql, operation, collectionName, batchSize = -1L, error = null)
            return res
        } catch (e: Throwable) {
            stopSpan(span, sql, operation, collectionName, batchSize = -1L, error = e)
            throw e
        }
    }

    protected inline fun <T> executeBatchWithTrace(sql: String?, block: () -> T): T {
        val operation = sql?.let { JdbcTraceKeywords.parseOperationAndCollection(it).first } ?: "BATCH"
        val collectionName = sql?.let { JdbcTraceKeywords.parseOperationAndCollection(it).second }
        val spanName = if (collectionName != null) "$operation $collectionName" else operation
        val span = tracer.startSpan(spanName)

        try {
            val res = block()
            val batchSize = when (res) {
                is IntArray -> res.size.toLong()
                is LongArray -> res.size.toLong()
                else -> 0L
            }
            stopSpan(span, sql, operation, collectionName, batchSize = batchSize, error = null)
            return res
        } catch (e: Throwable) {
            stopSpan(span, sql, operation, collectionName, batchSize = 0L, error = e)
            throw e
        }
    }

    protected fun stopSpan(
        span: NanoSpan,
        sql: String?,
        operation: String,
        collectionName: String?,
        batchSize: Long,
        error: Throwable?
    ) {
        val isError = error != null
        if (!span.sampled && !isError) {
            tracer.stop(span, StatusCode.STATUS_CODE_OK, SpanKind.SPAN_KIND_CLIENT, null)
            return
        }

        val attrs = HashMap<String, Any>(6)

        attrs[TraceUtil.ATTR_DB_SYSTEM_NAME] = traceProps.system
        attrs[TraceUtil.ATTR_SERVER_ADDRESS] = connectionInfo.host
        attrs[TraceUtil.ATTR_SERVER_PORT] = connectionInfo.port
        attrs[TraceUtil.ATTR_DB_OPERATION_NAME] = operation

        if (collectionName != null) attrs[TraceUtil.ATTR_DB_COLLECTION_NAME] = collectionName
        connectionInfo.namespace?.let { attrs[TraceUtil.ATTR_DB_NAMESPACE] = it }

        if (batchSize >= 0L) {
            attrs[TraceUtil.ATTR_DB_OPERATION_BATCH_SIZE] = batchSize
        }

        if (sql != null) {
            attrs[TraceUtil.ATTR_DB_QUERY_TEXT] = if (sql.length > traceProps.maxStatementLength) {
                sql.substring(0, traceProps.maxStatementLength) + traceProps.truncatedMarker
            } else sql
        }

        if (isError) {
            span.error = error
            if (error is SQLException) {
                val sqlState = error.sqlState
                if (!sqlState.isNullOrBlank()) {
                    attrs[TraceUtil.ATTR_DB_RESPONSE_STATUS_CODE] = sqlState
                    attrs[TraceUtil.ATTR_ERROR_TYPE] = sqlState
                } else {
                    attrs[TraceUtil.ATTR_ERROR_TYPE] = error.javaClass.name
                }
            } else {
                attrs[TraceUtil.ATTR_ERROR_TYPE] = error.javaClass.name
            }
        }

        tracer.stop(
            span = span,
            status = if (isError) StatusCode.STATUS_CODE_ERROR else StatusCode.STATUS_CODE_UNSET,
            kind = SpanKind.SPAN_KIND_CLIENT,
            attrs = attrs
        )
    }
}
