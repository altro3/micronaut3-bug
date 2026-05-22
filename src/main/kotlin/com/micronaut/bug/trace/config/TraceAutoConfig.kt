package com.micronaut.bug.trace.config

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.http.NanoTraceFilter
import com.micronaut.bug.trace.otlp.TraceBatcher
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.Ordered
import java.net.http.HttpClient
import java.util.concurrent.Executors

@ConditionalOnBooleanProperty("app.trace.enabled", matchIfMissing = true)
@Configuration
@EnableConfigurationProperties(TraceProperties::class)
class TraceAutoConfig {

    @Bean
    fun traceHttpClient(properties: TraceProperties) =
        HttpClient.newBuilder()
            .connectTimeout(properties.exporter.connectTimeout)
            .version(HttpClient.Version.HTTP_2)
            .executor(Executors.newFixedThreadPool(properties.exporter.maxSenders) { runnable ->
                Thread(runnable, "trace-http-sender").apply { isDaemon = true }
            })
            .build()

    @Bean
    fun traceBatcher(
        traceProps: TraceProperties,
        @Value($$"${spring.application.name}")
        appName: String,
        @Value($$"${app.node.name}")
        nodeName: String,
        @Qualifier("traceHttpClient")
        httpClient: HttpClient,
    ): TraceBatcher = TraceBatcher(
        appName = appName,
        nodeName = nodeName,
        traceProps = traceProps,
        httpClient = httpClient,
    )

    @Bean
    fun nanoTracer(traceBatcher: TraceBatcher, traceProps: TraceProperties) =
        NanoTracer(traceBatcher, traceProps)

    @Bean
    fun nanoTraceFilterRegistration(
        tracer: NanoTracer,
        traceProps: TraceProperties,
        @Value($$"${spring.application.name}") appName: String,
    ): FilterRegistrationBean<NanoTraceFilter> = FilterRegistrationBean(
        NanoTraceFilter(tracer, traceProps, appName)
    ).apply {
        order = Ordered.HIGHEST_PRECEDENCE
        addUrlPatterns("/*")
    }
}
