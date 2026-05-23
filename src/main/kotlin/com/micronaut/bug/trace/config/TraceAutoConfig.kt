package com.micronaut.bug.trace.config

import com.micronaut.bug.trace.NanoTracer
import com.micronaut.bug.trace.http.NanoTraceFilter
import com.micronaut.bug.trace.jdbc.JdbcTraceInterceptor
import com.micronaut.bug.trace.jdbc.TraceDataSource
import com.micronaut.bug.trace.otlp.TraceBatcher
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.beans.factory.annotation.Value
import org.springframework.beans.factory.config.BeanPostProcessor
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.core.Ordered
import org.springframework.core.env.Environment
import java.net.http.HttpClient
import java.util.concurrent.Executors
import javax.sql.DataSource

@ConditionalOnBooleanProperty("app.trace.enabled", matchIfMissing = true)
@AutoConfiguration
@EnableConfigurationProperties(TraceProperties::class)
class TraceAutoConfig {

    @Bean
    fun traceHttpClient(properties: TraceProperties) =
        HttpClient.newBuilder()
            .connectTimeout(properties.export.connectTimeout)
            .version(HttpClient.Version.HTTP_2)
            .executor(Executors.newFixedThreadPool(properties.export.maxSenders) { runnable ->
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

    @ConditionalOnBooleanProperty("app.trace.http.enabled", matchIfMissing = true)
    @Bean
    fun nanoTraceFilterRegistration(
        tracer: NanoTracer,
        traceProps: TraceProperties,
        @Value($$"${spring.application.name}") appName: String,
    ): FilterRegistrationBean<NanoTraceFilter> = FilterRegistrationBean(
        NanoTraceFilter(tracer, traceProps.http, appName)
    ).apply {
        order = Ordered.HIGHEST_PRECEDENCE
        addUrlPatterns("/*")
    }

    @ConditionalOnClass(DataSource::class)
    @ConditionalOnBooleanProperty("app.trace.jdbc.enabled", matchIfMissing = true)
    @Bean
    fun dataSourceTraceBeanPostProcessor(
        tracer: NanoTracer,
        traceProps: TraceProperties,
        environment: Environment
    ): BeanPostProcessor =
        object : BeanPostProcessor, Ordered {
            private val interceptor = JdbcTraceInterceptor(tracer, traceProps.jdbc, environment)

            override fun postProcessAfterInitialization(bean: Any, beanName: String): Any {
                if (bean is DataSource && bean !is TraceDataSource) {
                    return interceptor.wrap(bean)
                }
                return bean
            }

            override fun getOrder(): Int = Ordered.LOWEST_PRECEDENCE
        }

}
