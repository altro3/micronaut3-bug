package com.altro.common.log

import com.altro.common.log.LogConst.BODY_EMPTY
import com.altro.common.log.LogConst.BODY_TOO_LARGE
import com.altro.common.log.LogConst.STATUS_UNDEFINED
import com.altro.common.log.LogConst.STATUS_UNKNOWN
import com.altro.common.log.RequestWrapperFactory.wrap
import com.altro.common.log.RequestWrapperFactory.CachedBodyRequestWrapper
import com.altro.common.log.RequestWrapperFactory.MultipartTypeWrapper
import com.altro.common.log.config.LogProperties
import com.altro.common.trace.TraceUtil.MDC_COLOR
import com.altro.common.trace.TraceUtil.MDC_MAIN_STAT
import com.altro.common.trace.TraceUtil.MDC_SUB_TITLE
import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.http.HttpStatus
import org.springframework.util.ClassUtils
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingResponseWrapper

class ServerLoggingFilter(
    private val formatter: LogFormatter,
    private val logProps: LogProperties,
    private val logMasker: LogMasker? = null,
) : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}
    private val withActuator = ClassUtils.isPresent("org.springframework.boot.actuate.autoconfigure.endpoint.web.WebEndpointProperties", null)

    override fun doFilterInternal(rq: HttpServletRequest, rs: HttpServletResponse, chain: FilterChain) {
        val startTimeNano = System.nanoTime()

        if (withActuator && logProps.skipActuator && rq.requestURI.startsWith(rq.contextPath + PATH_ACTUATOR)) {
            chain.doFilter(rq, rs)
            return
        }

        val fullUri = getFullUri(rq)
        MDC.put(MDC_SUB_TITLE, "${rq.method} ${rq.requestURI}")
        val isDebugProvider = log.isDebugEnabled() && logProps.enabled

        val currentRq = wrap(rq, logProps.maxPayloadSize)

        val requestLogData by lazy {
            val headersMap = getHeadersMap(currentRq)
            val maskedHeaders = logMasker?.maskMap(headersMap) ?: headersMap

            val bodyResult = when (currentRq) {
                is CachedBodyRequestWrapper -> {
                    if (currentRq.body.isEmpty()) BODY_EMPTY
                    else formatter.formatBody(currentRq.body, currentRq.contentType, maskedHeaders, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                }
                is MultipartTypeWrapper -> formatter.formatMultipartResponse(
                    bytes = currentRq.parts.flatMap { (it as RequestWrapperFactory.ObservedPart).getContentBytes().asIterable() }.toByteArray(),
                    contentType = currentRq.contentType,
                    prettyPrint = logProps.prettyPrint,
                    limitLogSize = logProps.limitLogSize,
                    truncateChunkSize = logProps.truncateChunkSize
                )
                else -> BODY_TOO_LARGE
            }

            """
                |
                |================== Service request ==================
                |URI: ${currentRq.method} $fullUri
                |Headers: $maskedHeaders
                |Body:
                |$bodyResult
                |================== /Service request ==================
            """.trimMargin()
        }

        val rsWrapper = ContentCachingResponseWrapper(rs).apply {
            bufferSize = logProps.maxPayloadSize.toBytes().toInt()
        }

        try {
            if (isDebugProvider) log.debug { requestLogData }
            chain.doFilter(currentRq, rsWrapper)
        } finally {
            val durationNs = System.nanoTime() - startTimeNano
            val durationMs = durationNs / 1_000_000

            MDC.put(MDC_MAIN_STAT, "${durationMs}ms")
            val status = rsWrapper.status

            if (status >= 400) {
                MDC.put(MDC_COLOR, "red")
            } else if (durationMs >= 10_000) {
                MDC.put(MDC_COLOR, "yellow")
            }

            val isFullBodyCached = rsWrapper.contentSize < logProps.maxPayloadSize.toBytes()
            val rsBodyBytes = when {
                rsWrapper.contentSize == 0 -> BODY_EMPTY.toByteArray(Charsets.UTF_8)
                !isFullBodyCached -> BODY_TOO_LARGE.toByteArray(Charsets.UTF_8)
                else -> rsWrapper.contentAsByteArray
            }

            val responseLogData by lazy {
                val httpStatus = HttpStatus.resolve(status)
                val statusMessage = httpStatus?.reasonPhrase ?: if (status == 0) STATUS_UNDEFINED else STATUS_UNKNOWN
                val headersMap = getResponseHeaders(rsWrapper)
                val maskedHeaders = logMasker?.maskMap(headersMap) ?: headersMap

                val bodyText = if (ContentTypeAnalyzer.isMultipart(rsWrapper.contentType) && !rsBodyBytes.contentEquals(BODY_TOO_LARGE.toByteArray(Charsets.UTF_8))) {
                    formatter.formatMultipartResponse(rsBodyBytes, rsWrapper.contentType, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                } else {
                    formatter.formatBody(rsBodyBytes, rsWrapper.contentType, maskedHeaders, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                }

                """
                    |
                    |================== Service response ==================
                    |URI: ${currentRq.method} $fullUri
                    |Status: $status $statusMessage
                    |Duration: ${durationMs}ms
                    |Headers: $maskedHeaders
                    |Body:
                    |$bodyText
                    |================== /Service response ==================
                """.trimMargin()
            }

            if (status >= 400 && !isDebugProvider) {
                log.error { "Service failure detected!\n$requestLogData\n$responseLogData" }
            } else if (isDebugProvider) {
                log.debug { responseLogData }
            } else {
                log.info { "Service response success: ${currentRq.method} ${currentRq.requestURI} [$status]" }
            }

            rsWrapper.copyBodyToResponse()
            MDC.remove(MDC_SUB_TITLE)
            MDC.remove(MDC_MAIN_STAT)
            MDC.remove(MDC_COLOR)
        }
    }

    private fun getHeadersMap(rq: HttpServletRequest): Map<String, String> {
        val names = rq.headerNames ?: return emptyMap()
        val result = HashMap<String, String>()
        while (names.hasMoreElements()) {
            val name = names.nextElement()
            result[name] = rq.getHeader(name)
        }
        return result
    }

    private fun getResponseHeaders(rs: HttpServletResponse): Map<String, String> {
        val names = rs.headerNames ?: return emptyMap()
        val result = HashMap<String, String>()
        for (name in names) {
            result[name] = rs.getHeader(name)
        }
        return result
    }

    private fun getFullUri(rq: HttpServletRequest): String =
        rq.queryString?.let { "${rq.requestURI}?$it" } ?: rq.requestURI

    companion object {
        private const val PATH_ACTUATOR = "/actuator"
    }
}
