package com.micronaut.bug.client

import com.fasterxml.jackson.databind.ObjectMapper
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.ParameterizedTypeReference
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.http.converter.HttpMessageConverter
import org.springframework.retry.RetryCallback
import org.springframework.retry.support.RetryTemplate
import org.springframework.util.StopWatch
import org.springframework.web.client.ResponseErrorHandler
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClient.ResponseSpec
import com.micronaut.bug.client.HttpClientConst.HEADER_API_KEY
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_ON
import com.micronaut.bug.client.HttpClientUtils.createRestClient
import com.micronaut.bug.client.HttpClientUtils.createRetryTemplate

open class DefaultHttpClient {

    private val log = KotlinLogging.logger {}

    val restClient: RestClient
    val httpClientProperties: HttpClientProperties
    val retryTemplate: RetryTemplate?

    constructor(
        httpClientProperties: HttpClientProperties,
        restClient: RestClient,
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        this.restClient = restClient
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
        restClient: RestClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
        ),
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        this.restClient = restClient
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        objectMapper: ObjectMapper,
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        restClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            objectMapper = objectMapper,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
        )
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        restClientBuilder: RestClient.Builder,
        objectMapper: ObjectMapper,
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        restClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            clientBuilder = restClientBuilder,
            objectMapper = objectMapper,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
        )
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        restClientBuilder: RestClient.Builder,
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        restClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            clientBuilder = restClientBuilder,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
        )
        this.retryTemplate = retryTemplate
    }

    fun <Rs> sendRq(
        endpoint: Endpoint,
        responseClass: Class<Rs>,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
                    .body(responseClass)
            })
        } else {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
                .body(responseClass)
        }

    fun <Rs> sendRq(
        endpoint: Endpoint,
        responseType: ParameterizedTypeReference<Rs>,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
                    .body(responseType)
            })
        } else {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
                .body(responseType)
        }

    fun <Rs> sendRq(
        path: String,
        method: HttpMethod,
        responseClass: Class<Rs>,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers)
                    .body(responseClass)
            })
        } else {
            sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers)
                .body(responseClass)
        }

    fun <Rs> sendRq(
        path: String,
        method: HttpMethod,
        responseType: ParameterizedTypeReference<Rs>,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers)
                    .body(responseType)
            })
        } else {
            sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers)
                .body(responseType)
        }

    fun sendRq(
        endpoint: Endpoint,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): ResponseSpec =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
            })
        } else {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers)
        }

    fun sendRq(
        path: String,
        method: HttpMethod,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
    ): ResponseSpec {

        val rqBuilder = restClient.method(method)
        if (pathVars.isNullOrEmpty() && queryParams.isNullOrEmpty()) {
            rqBuilder.uri(path)
        } else {
            rqBuilder.uri(path) {
                queryParams?.forEach { param ->
                    it.queryParam(param.key, param.value)
                }
                return@uri if (!pathVars.isNullOrEmpty()) {
                    it.build(pathVars)
                } else {
                    it.build()
                }
            }
        }

        headers?.forEach { rqBuilder.header(it.key, it.value) }

        if (rqBody != null) {
            rqBuilder.body(rqBody)
            if (headers == null || !headers.containsKey(HttpHeaders.CONTENT_TYPE)) {
                rqBuilder.header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            }
        }

        if (httpClientProperties.type == HttpClientProperties.ClientType.INTERNAL && httpClientProperties.apiKey != null && withApiKey) {
            rqBuilder.header(HEADER_API_KEY, httpClientProperties.apiKey)
        }

        var stopWatch: StopWatch? = null
        if (log.isDebugEnabled() && httpClientProperties.requestTiming) {
            stopWatch = StopWatch()
            stopWatch.start()
        }

        val response = rqBuilder.retrieve()

        if (stopWatch != null) {
            stopWatch.stop()
            log.debug { "Request to $method $path was processed for ${stopWatch.totalTimeMillis}ms" }
        }
        return response
    }
}
