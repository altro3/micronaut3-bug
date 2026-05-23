package com.altro.common.trace.jdbc

import com.altro.common.trace.NanoTracer
import com.altro.common.trace.config.TraceProperties.TraceJdbcProperties
import com.altro.common.trace.jdbc.JdbcUrlParser.parse
import org.springframework.boot.jdbc.DataSourceUnwrapper
import org.springframework.core.env.Environment
import javax.sql.DataSource

class JdbcTraceInterceptor(
    private val tracer: NanoTracer,
    private val traceProps: TraceJdbcProperties,
    private val environment: Environment
) {

    fun wrap(dataSource: DataSource): DataSource {
        var jdbcUrl = environment.getProperty("spring.datasource.url")

        if (jdbcUrl.isNullOrBlank()) {
            jdbcUrl = try {
                val unwrapped = DataSourceUnwrapper.unwrap(dataSource, DataSource::class.java)
                unwrapped.connection.use { conn -> conn.metaData.url }
            } catch (_: Throwable) {
                null
            }
        }

        val connectionInfo = if (!jdbcUrl.isNullOrBlank()) {
            parse(jdbcUrl, traceProps.system)
        } else {
            JdbcUrlParser.DEFAULT_CONNECTION_INFO
        }

        return TraceDataSource(
            delegate = dataSource,
            tracer = tracer,
            traceProps = traceProps,
            connectionInfo = connectionInfo,
        )
    }
}
