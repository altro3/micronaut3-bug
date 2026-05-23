package com.micronaut.bug.trace.jdbc

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.config.TraceProperties.TraceJdbcProperties
import java.sql.Connection
import javax.sql.DataSource

@Suppress("JavaDefaultMethodsNotOverriddenByDelegation")
class TraceDataSource(
    private val delegate: DataSource,
    private val tracer: NanoTracer,
    private val traceProps: TraceJdbcProperties,
    private val serverAddress: String,
    private val serverPort: Long,
    private val dbNamespace: String?
) : DataSource by delegate {

    override fun getConnection(): Connection =
        TraceConnection(delegate.connection, tracer, traceProps, serverAddress, serverPort, dbNamespace)

    override fun getConnection(username: String?, password: String?): Connection =
        TraceConnection(delegate.getConnection(username, password), tracer, traceProps, serverAddress, serverPort, dbNamespace)
}
