package com.micronaut.bug.client

import com.fasterxml.jackson.databind.ObjectMapper
import com.micronaut.bug.client.HttpClientConst.HEADER_API_KEY
import com.micronaut.bug.client.HttpClientConst.HEADER_EXT_RQ_ID
import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_ON
import com.micronaut.bug.client.HttpClientUtils.createRestClient
import com.micronaut.bug.client.HttpClientUtils.createRetryTemplate
import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.ATTR_EXT_RQ_ID
import com.micronaut.bug.client.LoggingRequestInterceptor.Companion.ATTR_SKIP_LOGGING
import com.micronaut.bug.util.TraceIdGenerator
import org.springframework.core.ParameterizedTypeReference
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.http.converter.HttpMessageConverter
import org.springframework.retry.RetryCallback
import org.springframework.retry.support.RetryTemplate
import org.springframework.web.client.ResponseErrorHandler
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClient.ResponseSpec

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
        logBody: Boolean = true,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                    .body(responseClass)
            })
        } else {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                .body(responseClass)
        }

    fun <Rs> sendRq(
        endpoint: Endpoint,
        responseType: ParameterizedTypeReference<Rs>,
        rqBody: Any? = null,
        pathVars: Map<String, *>? = null,
        queryParams: Map<String, *>? = null,
        headers: Map<String, String>? = null,
        logBody: Boolean = true,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                    .body(responseType)
            })
        } else {
            sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
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
        logBody: Boolean = true,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                    .body(responseClass)
            })
        } else {
            sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers, logBody)
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
        logBody: Boolean = true,
    ): Rs? =
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(path, method, withApiKey, rqBody, pathVars, queryParams, headers, logBody)
                    .body(responseType)
            })
        } else {
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
        if (retryTemplate != null) {
            retryTemplate.execute(RetryCallback {
                sendRq(endpoint.path, endpoint.method, endpoint.withApiKey, rqBody, pathVars, queryParams, headers, logBody)
            })
        } else {
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
            if (headers == null || !headers.containsKey(CONTENT_TYPE)) {
                rqBuilder.header(CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            }
        }

        // Уникальный ID для связки конкретной пары запрос-ответ
        val extRqId = TraceIdGenerator.generate()
        rqBuilder.attribute(ATTR_EXT_RQ_ID, extRqId)

        if (httpClientProperties.type == INTERNAL) {
            if (httpClientProperties.apiKey != null && withApiKey) {
                rqBuilder.header(HEADER_API_KEY, httpClientProperties.apiKey)
            }
            rqBuilder.header(HEADER_EXT_RQ_ID, extRqId)
        }

        return rqBuilder.retrieve()
    }
}
