package com.micronaut.bug.trace.jdbc

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.config.TraceProperties.TraceJdbcProperties
import com.micronaut.bug.trace.jdbc.JdbcUrlParser.ConnectionInfo
import com.micronaut.bug.trace.jdbc.JdbcUrlParser.parse
import org.springframework.core.env.Environment
import javax.sql.DataSource

class JdbcTraceInterceptor(
    private val tracer: NanoTracer,
    private val traceProps: TraceJdbcProperties,
    private val environment: Environment
) {

    fun wrap(dataSource: DataSource): DataSource {
        // 1. Пытаемся взять URL из стандартной проперти Spring Boot
        var jdbcUrl = environment.getProperty("spring.datasource.url")

        // 2. Если в свойствах пусто, пробуем вытащить рефлексией из DataSource (например, HikariCP)
        if (jdbcUrl.isNullOrBlank()) {
            jdbcUrl = try {
                if (dataSource is com.zaxxer.hikari.HikariDataSource) {
                    dataSource.jdbcUrl
                } else null
            } catch (_: Throwable) {
                null
            }
        }

        // 3. Парсим URL один раз на старте
        val connectionInfo = if (!jdbcUrl.isNullOrBlank()) {
            parse(jdbcUrl, traceProps.system)
        } else {
            JdbcUrlParser.DEFAULT_CONNECTION_INFO
        }

        // 4. Передаем вычисленные статические сетевые параметры в обертку DataSource
        return TraceDataSource(
            delegate = dataSource,
            tracer = tracer,
            traceProps = traceProps,
            serverAddress = connectionInfo.host,
            serverPort = connectionInfo.port,
            dbNamespace = connectionInfo.namespace,
        )
    }
}
