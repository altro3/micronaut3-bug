package com.altro.common.client

import com.altro.common.client.HttpClientProperties.ClientType.EXTERNAL
import com.altro.common.client.NanoTraceClientInterceptor.Companion.PREFIX_CLIENT_SPAN
import com.altro.common.log.ContentTypeAnalyzer
import com.altro.common.log.LogConst.ATTR_EXT_RQ_ID
import com.altro.common.log.LogConst.ATTR_SKIP_LOGGING
import com.altro.common.log.LogConst.BODY_LOG_DISABLED
import com.altro.common.log.LogConst.BODY_TOO_LARGE
import com.altro.common.log.LogConst.ENCODING_GZIP
import com.altro.common.log.LogConst.MARKER_HTTP_PROTOCOL
import com.altro.common.log.LogConst.MDC_TARGET
import com.altro.common.log.LogConst.MDC_TYPE
import com.altro.common.log.LogConst.SLASH
import com.altro.common.log.LogConst.STRING_EMPTY
import com.altro.common.log.LogFormatter
import com.altro.common.log.LogMasker
import com.altro.common.trace.TraceIdGenerator.generateSpanId
import com.altro.common.trace.TraceUtil.MDC_COLOR
import com.altro.common.trace.TraceUtil.MDC_MAIN_STAT
import com.altro.common.trace.TraceUtil.MDC_SOURCE
import com.altro.common.trace.TraceUtil.MDC_SPAN_ID
import com.altro.common.trace.TraceUtil.MDC_SUB_TITLE
import io.github.oshai.kotlinlogging.KotlinLogging
import org.slf4j.MDC
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse
import java.io.ByteArrayInputStream
import java.util.zip.GZIPInputStream

class LoggingInterceptor(
    props: HttpClientProperties,
    private val formatter: LogFormatter,
    private val logMasker: LogMasker? = null,
) : ClientHttpRequestInterceptor {

    private val log = KotlinLogging.logger {}
    private val logProps = props.log
    private val serviceName = props.serviceName
    private val isExternal = props.type == EXTERNAL
    private val basePrefix: String = if (logProps.fullUrl) props.url.toString().removeSuffix(SLASH) else STRING_EMPTY

    override fun intercept(
        rq: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {
        val isDebug = log.isDebugEnabled() && logProps.enabled
        val skipLogging = rq.attributes[ATTR_SKIP_LOGGING] as? Boolean ?: false

        val extRqId = MDC.get(MDC_SPAN_ID) ?: generateSpanId()
        rq.attributes[ATTR_EXT_RQ_ID] = extRqId

        val urlFull = getFullUri(rq)
        val methodStr = rq.method.name()

        val oldClient = MDC.get(MDC_SOURCE)
        val oldServer = MDC.get(MDC_TARGET)
        val oldSubtitle = MDC.get(MDC_SUB_TITLE)
        val oldDuration = MDC.get(MDC_MAIN_STAT)
        val oldColor = MDC.get(MDC_COLOR)

        MDC.put(MDC_SOURCE, oldServer)
        MDC.put(MDC_TARGET, serviceName)
        MDC.put(MDC_SUB_TITLE, "$PREFIX_CLIENT_SPAN $methodStr $urlFull")
        if (isExternal) MDC.put(MDC_TYPE, EXTERNAL.name)
        MDC.put(MDC_COLOR, "green")

        val rqLogData by lazy {
            val maskedHeaders = logMasker?.maskMap(rq.headers.toSingleValueMap()) ?: rq.headers.toSingleValueMap()
            val ct = rq.headers.contentType?.toString()

            val bodyResult = when {
                skipLogging -> BODY_LOG_DISABLED
                body.size > logProps.maxPayloadSize.toBytes() -> BODY_TOO_LARGE
                ContentTypeAnalyzer.isMultipart(ct) -> formatter.formatMultipartResponse(body, ct, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                else -> formatter.formatBody(body, ct, maskedHeaders, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
            }

            """
                |
                |================== Client request ==================
                |URI: $methodStr $urlFull
                |ExtRqId: $extRqId
                |Headers: $maskedHeaders
                |Body:
                |$bodyResult
                |================== /Client request ==================
            """.trimMargin()
        }

        if (isDebug && !skipLogging) log.debug { rqLogData }
        val startTimeNano = System.nanoTime()

        try {
            val rs = try {
                execution.execute(rq, body)
            } catch (e: Exception) {
                val durationNs = System.nanoTime() - startTimeNano
                MDC.put(MDC_MAIN_STAT, "${durationNs / 1_000_000}ms")
                MDC.put(MDC_COLOR, "red")

                log.error(e) {
                    "External call failed! [extRqId: $extRqId, duration: ${durationNs}ns]" +
                            if (isDebug || skipLogging) "" else "\n$rqLogData"
                }
                throw e
            }

            val durationNs = System.nanoTime() - startTimeNano
            val durationMs = durationNs / 1_000_000
            val isError = rs.statusCode.isError
            val isSlow = durationMs >= logProps.slowThreshold.toMillis()

            when {
                isError -> MDC.put(MDC_COLOR, "red")
                isSlow -> MDC.put(MDC_COLOR, "yellow")
            }
            MDC.put(MDC_MAIN_STAT, "${durationMs}ms")

            if (isDebug || isError) {
                val rsBodyBytes = extractResponseBody(rs, skipLogging)

                val rsLogData = run {
                    val maskedHeaders = logMasker?.maskMap(rs.headers.toSingleValueMap()) ?: rs.headers.toSingleValueMap()
                    val ct = rs.headers.contentType?.toString()

                    val bodyResult = when {
                        skipLogging -> BODY_LOG_DISABLED
                        rsBodyBytes.contentEquals(BODY_TOO_LARGE.toByteArray()) -> BODY_TOO_LARGE
                        ContentTypeAnalyzer.isMultipart(ct) -> formatter.formatMultipartResponse(rsBodyBytes, ct, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                        else -> formatter.formatBody(rsBodyBytes, ct, maskedHeaders, logProps.prettyPrint, logProps.limitLogSize, logProps.truncateChunkSize)
                    }

                    """
                        |
                        |================== Client response ==================
                        |URI: $methodStr $urlFull
                        |ExtRqId: $extRqId
                        |Status: ${rs.statusCode.value()} (${rs.statusText})
                        |Duration: ${durationMs}ms
                        |Headers: $maskedHeaders
                        |Body:
                        |$bodyResult
                        |================== /Client response ==================
                    """.trimMargin()
                }

                if (isError && !isDebug && !skipLogging) {
                    log.error { "External service failure!\n$rqLogData\n$rsLogData" }
                } else if (!skipLogging) {
                    log.debug { rsLogData }
                }
            }
            return rs
        } finally {
            MDC.put(MDC_SOURCE, oldClient)
            MDC.put(MDC_TARGET, oldServer)
            MDC.put(MDC_SUB_TITLE, oldSubtitle)
            MDC.put(MDC_MAIN_STAT, oldDuration)
            MDC.put(MDC_COLOR, oldColor)
            MDC.remove(MDC_TYPE)
        }
    }

    private fun extractResponseBody(rs: ClientHttpResponse, skipLogging: Boolean): ByteArray {
        if (skipLogging) return BODY_LOG_DISABLED.toByteArray()

        val maxAllowed = logProps.maxPayloadSize.toBytes()
        val contentLength = rs.headers.contentLength

        if (contentLength > maxAllowed) return BODY_TOO_LARGE.toByteArray()

        val rawBytes = rs.body.readAllBytes()
        if (rawBytes.size > maxAllowed) return BODY_TOO_LARGE.toByteArray()

        return decompressIfNeeded(rs, rawBytes)
    }

    private fun decompressIfNeeded(response: ClientHttpResponse, bytes: ByteArray): ByteArray {
        val encoding = response.headers.getFirst(HttpHeaders.CONTENT_ENCODING)
        val isGzip = encoding?.contains(ENCODING_GZIP, ignoreCase = true) == true

        if (!isGzip || bytes.isEmpty()) return bytes

        return runCatching {
            GZIPInputStream(ByteArrayInputStream(bytes)).use { it.readBytes() }
        }.getOrElse { e ->
            log.warn { "Failed to decompress GZIP body, logging raw data. Error: ${e.message}" }
            bytes
        }
    }

    private fun getFullUri(rq: HttpRequest): String {
        val path = rq.uri.toString()
        return if (basePrefix.isNotEmpty() && !path.startsWith(MARKER_HTTP_PROTOCOL)) {
            basePrefix + path
        } else {
            path
        }
    }
}
