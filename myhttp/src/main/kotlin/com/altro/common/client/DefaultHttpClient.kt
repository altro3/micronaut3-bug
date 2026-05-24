package com.altro.common.client

import com.altro.common.client.HttpClientConst.HEADER_API_KEY
import com.altro.common.client.HttpClientProperties.ClientType.INTERNAL
import com.altro.common.client.HttpClientUtils.DEFAULT_RETRY_ON
import com.altro.common.client.HttpClientUtils.createRestClient
import com.altro.common.client.HttpClientUtils.createRetryTemplate
import com.altro.common.log.LogConst.ATTR_SKIP_LOGGING
import com.altro.common.trace.NanoTracer
import com.altro.common.trace.TraceUtil.METHODS_WITHOUT_BODY
import org.springframework.core.ParameterizedTypeReference
import org.springframework.core.retry.RetryTemplate
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.http.converter.HttpMessageConverter
import org.springframework.web.client.ResponseErrorHandler
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClient.ResponseSpec
import tools.jackson.databind.json.JsonMapper

open class DefaultHttpClient {

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
        tracer: NanoTracer? = null,
        restClient: RestClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
            tracer = tracer,
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
        jsonMapper: JsonMapper,
        tracer: NanoTracer? = null,
        messageConverters: List<HttpMessageConverter<*>>? = null,
        errorHandler: ResponseErrorHandler? = null,
        retryOn: List<Class<out Throwable>>? = null,
        retryTemplate: RetryTemplate? = createRetryTemplate(httpClientProperties, retryOn ?: DEFAULT_RETRY_ON),
    ) {
        this.httpClientProperties = httpClientProperties
        restClient = createRestClient(
            senderAppName = senderAppName,
            clientProps = httpClientProperties,
            jsonMapper = jsonMapper,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
            tracer = tracer,
        )
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        restClientBuilder: RestClient.Builder,
        jsonMapper: JsonMapper,
        tracer: NanoTracer? = null,
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
            jsonMapper = jsonMapper,
            messageConverters = messageConverters,
            errorHandler = errorHandler,
            tracer = tracer,
        )
        this.retryTemplate = retryTemplate
    }

    constructor(
        senderAppName: String,
        httpClientProperties: HttpClientProperties,
        restClientBuilder: RestClient.Builder,
        tracer: NanoTracer? = null,
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
            tracer = tracer,
        )
        this.retryTemplate = retryTemplate
    }

    fun <Rs : Any> sendRq(
        endpoint: Endpoint,
        responseClass: Class<Rs>,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): Rs? =
        executeWithRetry {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                .body(responseClass)
        }

    fun <Rs : Any> sendRq(
        endpoint: Endpoint,
        responseType: ParameterizedTypeReference<Rs>,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): Rs? =
        executeWithRetry {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                .body(responseType)
        }

    fun <Rs : Any> sendRq(
        path: String,
        method: HttpMethod,
        responseClass: Class<Rs>,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): Rs? =
        executeWithRetry {
            sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                .body(responseClass)
        }

    fun <Rs : Any> sendRq(
        path: String,
        method: HttpMethod,
        responseType: ParameterizedTypeReference<Rs>,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): Rs? =
        executeWithRetry {
            sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                .body(responseType)
        }

    fun sendRq(
        endpoint: Endpoint,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): ResponseSpec =
        executeWithRetry {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
        }

    fun sendRq(
        path: String,
        method: HttpMethod,
        withApiKey: Boolean = false,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): ResponseSpec {

        val rqBuilder = restClient.method(method)

        // Установка атрибута для управления логированием в интерцепторе
        rqBuilder.attribute(ATTR_SKIP_LOGGING, !logBody)

        if (pathVars.isNullOrEmpty() && queryParams.isNullOrEmpty()) {
            rqBuilder.uri(path)
        } else {
            rqBuilder.uri(path) { uriVars ->
                queryParams?.forEach { param ->
                    param.value?.let { uriVars.queryParam(param.key, it) }
                }
                if (!pathVars.isNullOrEmpty()) {
                    uriVars.build(pathVars)
                } else {
                    uriVars.build()
                }
            }
        }

        headers?.forEach { rqBuilder.header(it.key, it.value) }

        if (rqBody != null) {
            rqBuilder.body(rqBody)
            if (method.name() !in METHODS_WITHOUT_BODY && (headers == null || !headers.containsKey(CONTENT_TYPE))) {
                rqBuilder.header(CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            }
        }

        if (httpClientProperties.type == INTERNAL) {
            if (withApiKey) {
                httpClientProperties.apiKey?.let { rqBuilder.header(HEADER_API_KEY, it) }
            }
        }

        return rqBuilder.retrieve()
    }

    private fun <T> executeWithRetry(block: () -> T): T {
        return if (retryTemplate != null) {
            retryTemplate.invoke<T> { block() }
        } else {
            block()
        }
    }
}
