package com.micronaut.bug.config.trace.config

import com.micronaut.bug.config.trace.NanoTraceFilter
import com.micronaut.bug.config.trace.NanoTracer
import com.micronaut.bug.config.trace.TempoExporter
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.Ordered
import java.net.http.HttpClient

@ConditionalOnBooleanProperty("app.trace.enabled", matchIfMissing = true)
@Configuration
@EnableConfigurationProperties(TraceProperties::class)
class TraceAutoConfig {

    @Bean
    fun traceHttpClient(properties: TraceProperties): HttpClient =
        HttpClient.newBuilder()
            .connectTimeout(properties.exporter.connectTimeout)
            .build()

    @Bean
    fun tempoExporter(
        traceProps: TraceProperties,
        @Value("\${spring.application.name}")
        appName: String,
        @Qualifier("traceHttpClient")
        httpClient: HttpClient,
    ) = TempoExporter(
        appName = appName,
        httpClient = httpClient,
        traceProps = traceProps,
    )

    @Bean
    fun nanoTracer(tempoExporter: TempoExporter) =
        NanoTracer(tempoExporter)

    // Регистрация серверного фильтра трассировки
    @Bean
    fun nanoTraceFilterRegistration(
        tracer: NanoTracer,
        @Value("\${spring.application.name}")
        appName: String,
    ) = FilterRegistrationBean(NanoTraceFilter(tracer, appName)).apply {
        // Трейсинг ВСЕГДА идет первым
        order = Ordered.HIGHEST_PRECEDENCE
    }
}
