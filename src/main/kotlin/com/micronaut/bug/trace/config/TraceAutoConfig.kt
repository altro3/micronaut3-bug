package com.micronaut.bug.trace.config

import com.micronaut.bug.trace.NanoTraceFilter
import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.otlp.TraceExporter
import io.micrometer.observation.ObservationPredicate
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.Ordered
import org.springframework.http.server.observation.ServerRequestObservationContext
import java.net.http.HttpClient

@ConditionalOnBooleanProperty("app.trace.enabled", matchIfMissing = true)
@Configuration
@EnableConfigurationProperties(TraceProperties::class)
class TraceAutoConfig {

    @Bean
    fun traceHttpClient(properties: TraceProperties) =
        HttpClient.newBuilder()
            .connectTimeout(properties.exporter.connectTimeout)
            .build()

    @Bean
    fun traceExporter(
        traceProps: TraceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        @Value($$"${app.node.name}")
        nodeName: String,
        @Qualifier("traceHttpClient")
        httpClient: HttpClient,
    ) = TraceExporter(
        appName = appName,
        nodeName = nodeName,
        httpClient = httpClient,
        traceProps = traceProps,
    )

    @Bean
    fun nanoTracer(traceExporter: TraceExporter, traceProps: TraceProperties) =
        NanoTracer(traceExporter, traceProps)

    // Регистрация серверного фильтра трассировки
    @Bean
    fun nanoTraceFilterRegistration(
        tracer: NanoTracer,
        traceProps: TraceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
    ) = FilterRegistrationBean(NanoTraceFilter(tracer, traceProps, appName)).apply {
        // Трейсинг ВСЕГДА идет первым
        order = Ordered.HIGHEST_PRECEDENCE
    }
}
