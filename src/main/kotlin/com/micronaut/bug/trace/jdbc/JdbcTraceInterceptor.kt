package com.micronaut.bug.trace.jdbc

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.config.TraceProperties.TraceJdbcProperties
import javax.sql.DataSource

class JdbcTraceInterceptor(
    private val tracer: NanoTracer,
    private val traceProps: TraceJdbcProperties
) {

    fun wrap(dataSource: DataSource): DataSource =
        TraceDataSource(dataSource, tracer, traceProps)
}
