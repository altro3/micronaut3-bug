package com.micronaut.bug.trace.gateway

//import org.springframework.cloud.gateway.filter.GatewayFilterChain
//import org.springframework.cloud.gateway.filter.GlobalFilter

/*
@Component
class NanoTraceGatewayFilter(
    private val tracer: NanoTracer,
    private val traceProps: TraceProperties,
    @Value($$"${spring.application.name}")
    private val selfServiceName: String,
) : GlobalFilter, Ordered {

    private val log = KotlinLogging.logger {}

    override fun filter(exchange: ServerWebExchange, chain: GatewayFilterChain): Mono<Void> {
        val rq = exchange.request
        val startTimeNano = System.nanoTime()

        // 1. Извлекаем данные из заголовков (W3C)
        val traceParent = rq.headers.getFirst(HEADER_TRACEPARENT)
        var traceId: String? = null
        var parentId: String? = null
        var isSampledByParent = true

        if (traceParent != null && traceParent.startsWith(TRACEPARENT_PREFIX)) {
            val parts = traceParent.split(TRACEPARENT_DELIMITER)
            if (parts.size >= 4) {
                traceId = parts[1]
                parentId = parts[2]
                val flags = parts[3].toIntOrNull(16) ?: 0
                isSampledByParent = (flags and 0x01) == 1
            }
        }

        val sender = rq.headers.getFirst(HEADER_X_SENDER) ?: DEFAULT_SENDER
        val baggage = parseBaggage(rq.headers.getFirst(HEADER_BAGGAGE))

        val propagationHeaders = traceProps.propagationHeaders
            .mapNotNull { headerName ->
                rq.headers.getFirst(headerName)?.let { headerName.lowercase() to it }
            }
            .toMap()

        val traceState = rq.headers.getFirst(NanoTracer.HEADER_TRACESTATE)

        // 2. Стартуем трейс
        val ctx = tracer.startTrace(
            name = "GW ${rq.method} ${rq.path}",
            remoteTraceId = traceId,
            remoteParentId = parentId,
            sampled = isSampledByParent,
            baggage = baggage,
            propagationHeaders = propagationHeaders,
            traceState = traceState,
        )

        // 3. Мутируем запрос (прокидываем traceId дальше в микросервисы)
        val mutatedRequest = exchange.request.mutate()
            .header(HEADER_TRACEPARENT, tracer.getTraceParent(ctx))
            .header(HEADER_X_SENDER, selfServiceName)
            .build()

        return chain.filter(exchange.mutate().request(mutatedRequest).build())
            .contextWrite { it.put(TraceContext::class.java, ctx) } // Сохраняем в реактивный контекст
            .doFinally { signalType ->
                tracer.syncMdc(ctx)
                val rs = exchange.response
                val durationMs = (System.nanoTime() - startTimeNano) / 1_000_000
                val isSlow = durationMs >= traceProps.slowRequestThreshold.toMillis()

                if (isSlow) {
                    log.warn { "Slow request at Gateway: ${rq.method} ${rq.path} took ${durationMs}ms" }
                }

                val isError = rs.statusCode?.isError == true || signalType == SignalType.ON_ERROR

                val attrs = buildMap<String, Any> {
                    put(ATTR_URL_FULL, rq.uri.toString())
                    put(ATTR_URL_SCHEME, rq.uri.scheme)
                    put(ATTR_URL_PATH, rq.path.toString())
                    rq.uri.query?.let { put(ATTR_URL_QUERY, it) }

                    put(ATTR_HTTP_REQUEST_METHOD, rq.method.name())
                    rq.headers.getFirst(HttpHeaders.USER_AGENT)?.let { put(ATTR_USER_AGENT_ORIGINAL, it) }

                    // Content-Length в реактивном запросе
                    if (rq.method.name() !in METHODS_WITHOUT_BODY) {
                        val reqSize = rq.headers.contentLength
                        if (reqSize != -1L) {
                            put(ATTR_HTTP_REQUEST_BODY_SIZE, reqSize)
                        }
                    }

                    if (isSlow) put(ATTR_HTTP_SLOW_REQUEST, true)

                    // Content-Length ответа
                    rs.headers.contentLength.takeIf { it != -1L }?.let {
                        put(ATTR_HTTP_RESPONSE_BODY_SIZE, it)
                    }

                    put(ATTR_CLIENT, sender)
                    put(ATTR_SERVER, selfServiceName)

                    put(ATTR_SERVER_ADDRESS, rq.uri.host ?: "unknown")
                    put(ATTR_SERVER_PORT, rq.uri.port.toLong())
                    rq.remoteAddress?.let { put(ATTR_CLIENT_ADDRESS, it.address.hostAddress) }

                    rs.statusCode?.let { put(ATTR_HTTP_RESPONSE_STATUS_CODE, it.value().toLong()) }

                    if (isError) {
                        put(ATTR_EXCEPTION_MESSAGE, "HTTP ${rs.statusCode?.value() ?: "UNKNOWN"}")
                    }

                    putAll(getRequestHeadersAttrs(rq))
                    putAll(getResponseHeadersAttrs(rs))
                }

                tracer.stop(
                    ctx = ctx,
                    status = if (isError) Status.StatusCode.STATUS_CODE_ERROR else Status.StatusCode.STATUS_CODE_OK,
                    attrs = attrs,
                    forceExport = isSlow || isError,
                )
                MDC.clear()
            }
    }

    private fun getRequestHeadersAttrs(rq: ServerHttpRequest): Map<String, List<String>> =
        rq.headers.keys.asSequence()
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
                val normalizedName = name.lowercase()
                val key = "${PREFIX_HTTP_REQUEST_HEADER}$normalizedName"
                key to if (normalizedName in SENSITIVE_HEADERS) MASKED_VALUES else (rq.headers[name] ?: emptyList())
            }

    private fun getResponseHeadersAttrs(rs: ServerHttpResponse): Map<String, List<String>> =
        rs.headers.keys.asSequence()
            .filter { it.lowercase() !in OTEL_MAPPED_HEADERS }
            .associate { name ->
                val key = "${PREFIX_HTTP_RESPONSE_HEADER}${name.lowercase()}"
                key to (rs.headers[name] ?: emptyList())
            }

    override fun getOrder(): Int = Ordered.HIGHEST_PRECEDENCE

    companion object {
        private const val DEFAULT_SENDER = "USER"
    }
}
*/
