package com.micronaut.bug.trace.reactor

/**
 * Реактивный фильтр для [org.springframework.web.reactive.function.client.WebClient].
 * Аналог [com.micronaut.bug.client.NanoTraceClientInterceptor] для неблокирующего стека.
 */
/*
class NanoTraceReactiveClientFilter(
    private val tracer: NanoTracer,
    private val selfServiceName: String,
    private val clientProps: HttpClientProperties,
) : ExchangeFilterFunction {

    override fun filter(request: ClientRequest, next: RestClient.RequestHeadersSpec.ExchangeFunction): Mono<ClientResponse> {
        return Mono.deferContextual { reactorCtx ->
            val startTimeNano = System.nanoTime()

            // 1. Пытаемся достать контекст из Reactor Context (проброшен Гейтвеем или контроллером)
            val parentCtx = tracer.currentContext(reactorCtx)

            // 2. Стартуем дочерний спан
            val urlFull = request.url().toString()
            val ctx = tracer.startSpan(
                name = "${request.method()} $urlFull",
                // Явно передаем родителя, так как ThreadLocal тут пуст
                parent = parentCtx
            )

            // 3. Формируем новый запрос с проброшенными заголовками
            val mutatedRequest = ClientRequest.from(request)
                .header(NanoTracer.HEADER_TRACEPARENT, tracer.getTraceParent(ctx))
                .apply {
                    if (clientProps.type == HttpClientProperties.ClientType.INTERNAL) {
                        MessageSupport.header(NanoTracer.HEADER_X_SENDER, selfServiceName)
                    }
                    NanoTracer.formatBaggage(ctx.baggage)?.let { MessageSupport.header(NanoTracer.HEADER_BAGGAGE, it) }
                    ctx.traceState?.let { MessageSupport.header(NanoTracer.HEADER_TRACESTATE, it) }

                    // Проброс кастомных propagationHeaders
                    ctx.propagationHeaders.forEach { (name, value) ->
                        if (!request.headers().containsKey(name)) {
                            MessageSupport.header(name, value)
                        }
                    }
                }
                .build()

            next.exchange(mutatedRequest)
                .doOnNext { response ->
                    val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
                    val isSlow = durationMs >= clientProps.log.slowThreshold.toMillis()
                    val isError = response.statusCode().isError

                    tracer.stop(
                        ctx = ctx,
                        status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                        kind = Span.SpanKind.SPAN_KIND_CLIENT,
                        forceExport = isSlow || isError,
                        attrs = buildMap {
                            putAll(getBaseAttrs(mutatedRequest, response, isError))
                            if (isSlow) put(NanoTracer.ATTR_HTTP_SLOW_REQUEST, true)
                        }
                    )
                }
                .doOnError { e ->
                    // Сетевые ошибки (Timeout, Connection Refused)
                    tracer.stop(
                        ctx = ctx,
                        status = Status.StatusCode.STATUS_CODE_ERROR,
                        kind = Span.SpanKind.SPAN_KIND_CLIENT,
                        attrs = buildMap {
                            put(NanoTracer.ATTR_HTTP_REQUEST_METHOD, request.method().name())
                            put(NanoTracer.ATTR_URL_FULL, urlFull)
                            put(NanoTracer.ATTR_EXCEPTION_TYPE, e.javaClass.name)
                            put(NanoTracer.ATTR_EXCEPTION_MESSAGE, e.message ?: e.javaClass.simpleName)
                            put(NanoTracer.ATTR_EXCEPTION_STACKTRACE, e.stackTraceToString())
                        }
                    )
                }
        }
    }

    private fun getBaseAttrs(rq: ClientRequest, rs: ClientResponse, isError: Boolean) = buildMap<String, Any> {
        put(NanoTracer.ATTR_HTTP_REQUEST_METHOD, rq.method().name())
        put(NanoTracer.ATTR_URL_FULL, rq.url().toString())
        put(NanoTracer.ATTR_HTTP_RESPONSE_STATUS_CODE, rs.statusCode().value().toLong())

        val host = rq.url().host ?: clientProps.url.host ?: "unknown"
        put(NanoTracer.ATTR_SERVER_ADDRESS, host)

        val port = rq.url().port.takeIf { it != -1 } ?: clientProps.url.port.toLong()
        put(NanoTracer.ATTR_SERVER_PORT, port)

        put(NanoTracer.ATTR_CLIENT, selfServiceName)
        put(NanoTracer.ATTR_SERVER, clientProps.serviceName.toString())
        put(NanoTracer.ATTR_PEER_SERVICE, clientProps.serviceName.toString())

        if (rq.method().name() !in NanoTracer.METHODS_WITHOUT_BODY) {
            // В WebClient мы не можем легко получить размер ByteArray тела без подписки,
            // поэтому берем из заголовков, если там есть Content-Length
            rq.headers().contentLength.takeIf { it != -1L }?.let {
                put(NanoTracer.ATTR_HTTP_REQUEST_BODY_SIZE, it)
            }
        }

        rs.headers().contentLength().takeIf { it.isPresent }?.let {
            put(NanoTracer.ATTR_HTTP_RESPONSE_BODY_SIZE, it.asLong)
        }

        if (isError) {
            put(NanoTracer.ATTR_EXCEPTION_MESSAGE, "HTTP ${rs.statusCode().value()}")
        }

        // Заголовки (с фильтрацией и маскированием)
        putAll(extractHeaders(rq.headers(), NanoTracer.PREFIX_HTTP_REQUEST_HEADER))
        putAll(extractHeaders(rs.headers().asHttpHeaders(), NanoTracer.PREFIX_HTTP_RESPONSE_HEADER))
    }

    private fun extractHeaders(headers: HttpHeaders, prefix: String) =
        headers.keys.asSequence()
            .filter { it.lowercase() !in NanoTracer.OTEL_MAPPED_HEADERS }
            .associate { name ->
                val key = "$prefix${name.lowercase()}"
                key to (headers[name] ?: emptyList())
            }
}*/
