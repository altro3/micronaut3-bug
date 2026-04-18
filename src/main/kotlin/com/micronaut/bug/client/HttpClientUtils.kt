package com.micronaut.bug.client

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.json.JsonMapper
import io.github.oshai.kotlinlogging.KotlinLogging
import io.netty.channel.ChannelOption
import io.netty.handler.timeout.ReadTimeoutHandler
import org.springframework.http.client.BufferingClientHttpRequestFactory
import org.springframework.http.client.ClientHttpRequestFactory
import org.springframework.http.client.ReactorClientHttpRequestFactory
import org.springframework.http.converter.ByteArrayHttpMessageConverter
import org.springframework.http.converter.HttpMessageConverter
import org.springframework.http.converter.ResourceHttpMessageConverter
import org.springframework.http.converter.StringHttpMessageConverter
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter
import org.springframework.http.converter.support.AllEncompassingFormHttpMessageConverter
import org.springframework.retry.support.RetryTemplate
import org.springframework.web.client.HttpServerErrorException
import org.springframework.web.client.ResourceAccessException
import org.springframework.web.client.ResponseErrorHandler
import org.springframework.web.client.RestClient
import reactor.netty.http.client.HttpClient
import reactor.netty.transport.ProxyProvider
import com.micronaut.bug.client.HttpClientConst.HEADER_SENDER_APP_NAME
import java.time.Duration
import java.util.concurrent.TimeUnit

object HttpClientUtils {

    private val log = KotlinLogging.logger {}

    val DEFAULT_READ_TIMEOUT: Duration = Duration.ofSeconds(30)
    val DEFAULT_CONNECT_TIMEOUT: Duration = Duration.ofSeconds(10)
    const val DEFAULT_MAX_ATTEMPTS: Int = 1
    val DEFAULT_RETRY_DELAY: Duration = Duration.ofSeconds(1)
    const val DEFAULT_RETRY_MULTIPLIER: Double = 1.0
    val DEFAULT_RETRY_ON = listOf<Class<out Throwable>>(
        ResourceAccessException::class.java,
        HttpServerErrorException::class.java,
    )

    fun createRestClient(
        senderAppName: String,
        clientProps: HttpClientProperties,
        clientBuilder: RestClient.Builder = RestClient.builder(),
        objectMapper: ObjectMapper = JsonMapper.builder().build(),
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
    ): RestClient {
        val requestFactory = createRequestFactory(clientProps)
        val builder = clientBuilder
            .messageConverters(
                messageConverters ?: listOf(
                    MappingJackson2HttpMessageConverter(objectMapper),
                    StringHttpMessageConverter(Charsets.UTF_8),
                    ByteArrayHttpMessageConverter(),
                    ResourceHttpMessageConverter(false),
                    AllEncompassingFormHttpMessageConverter(),
                )
            )

        builder.observationConvention(HttpClientObservationConvention(clientProps))

        if (errorHandler != null) {
            builder.defaultStatusHandler(errorHandler)
        }

        if (clientProps.logging) {
            // ВАЖНО: BufferingClientHttpRequestFactory позволяет перечитывать InputStream тела.
            // Без него интерцептор логирования "съест" данные, и клиент получит пустое тело.
            // Добавляем наш интерцептор
            builder.requestFactory(BufferingClientHttpRequestFactory(requestFactory))
                .requestInterceptor(LoggingRequestInterceptor(clientProps))
        } else {
            builder.requestFactory(requestFactory)
        }

        if (clientProps.tracing) {
            builder.requestInterceptor(TracingForwardingInterceptor())
        }

        builder.defaultHeader(HEADER_SENDER_APP_NAME, senderAppName)

        return builder.build()
    }

    fun createRequestFactory(clientProperties: HttpClientProperties): ClientHttpRequestFactory =
        ReactorClientHttpRequestFactory(createHttpClient(clientProperties))

    fun createHttpClient(clientProperties: HttpClientProperties): HttpClient {

        var httpClient = HttpClient.create()
            .baseUrl(clientProperties.url.toString())
            .responseTimeout(clientProperties.readTimeout)
            .doOnConnected { it.addHandlerLast(ReadTimeoutHandler(clientProperties.readTimeout.seconds, TimeUnit.SECONDS)) }
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, clientProperties.connectTimeout.toMillis().toInt())

        val proxyProps = clientProperties.proxy
        if (proxyProps != null && proxyProps.enabled) {
            log.info { "Using Proxy request factory for url ${clientProperties.url}" }
            httpClient = httpClient.proxy { typeSpec ->
                typeSpec.type(ProxyProvider.Proxy.HTTP)
                (typeSpec as ProxyProvider.AddressSpec)
                    .host(proxyProps.host)
                    .port(proxyProps.port)
                    .connectTimeoutMillis(proxyProps.connectTimeout.toMillis())

                if (proxyProps.username != null) {
                    (typeSpec as ProxyProvider.Builder)
                        .username(proxyProps.username!!)
                        .password { proxyProps.password }
                }
            }
        }

        return httpClient
    }

    fun createRetryTemplate(clientProps: HttpClientProperties, retryOn: List<Class<out Throwable>>): RetryTemplate? {

        if (clientProps.maxAttempts == DEFAULT_MAX_ATTEMPTS) {
            return null
        }

        val builder = RetryTemplate.builder()
        if (clientProps.maxAttempts > 0) {
            builder.maxAttempts(clientProps.maxAttempts)
        } else {
            builder.infiniteRetry()
        }

        val backoffProps = clientProps.retryBackoff
        if (backoffProps.multiplier == DEFAULT_RETRY_MULTIPLIER) {
            builder.fixedBackoff(backoffProps.delay)
        } else {
            require(backoffProps.maxDelay != null && backoffProps.maxDelay > backoffProps.delay) {
                "You use multiplier, need to set maxDelay. And maxDelay must be much then delay"
            }
            builder.exponentialBackoff(backoffProps.delay, backoffProps.multiplier, backoffProps.maxDelay, backoffProps.random)
        }

        return builder
            .retryOn(retryOn)
            .build()
    }
}
