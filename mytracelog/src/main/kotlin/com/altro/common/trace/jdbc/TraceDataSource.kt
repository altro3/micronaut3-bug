package com.altro.common.trace.jdbc

import com.altro.common.trace.NanoTracer
import com.altro.common.trace.config.TraceProperties.TraceJdbcProperties
import com.altro.common.trace.jdbc.JdbcUrlParser.ConnectionInfo
import java.sql.Connection
import javax.sql.DataSource

@Suppress("JavaDefaultMethodsNotOverriddenByDelegation")
class TraceDataSource(
    private val delegate: DataSource,
    private val tracer: NanoTracer,
    private val traceProps: TraceJdbcProperties,
    private val connectionInfo: ConnectionInfo,
) : DataSource by delegate {

    override fun getConnection(): Connection =
        TraceConnection(delegate.connection, tracer, traceProps, connectionInfo)

    override fun getConnection(username: String?, password: String?): Connection =
        TraceConnection(delegate.getConnection(username, password), tracer, traceProps, connectionInfo)
}
