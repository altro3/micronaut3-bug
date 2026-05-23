package com.altro.common.trace.otlp

import com.altro.common.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.delay
import kotlinx.coroutines.future.await
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpHeaders.CONTENT_ENCODING
import org.springframework.http.MediaType
import java.io.ByteArrayOutputStream
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.ThreadLocalRandom
import java.util.zip.GZIPOutputStream
import kotlin.math.pow
import kotlin.time.Duration.Companion.milliseconds

class OtlpHttpSender(
    appName: String,
    nodeName: String,
    traceProps: TraceProperties,
    private val httpClient: HttpClient,
    private val eventPool: ConcurrentLinkedQueue<TraceEvent>
) {
    private val log = KotlinLogging.logger {}

    private val exporterProps = traceProps.export
    private val retryInterval = exporterProps.retryInterval.toMillis()
    private val maxAttempts = exporterProps.maxAttempts
    private val encoder = OtlpTraceEncoder(appName, nodeName, traceProps)

    suspend fun sendBatch(events: List<TraceEvent>) {
        if (events.isEmpty()) return

        try {
            val rawPayload = encoder.encodeBatch(events)
            var payload = rawPayload

            val shouldCompress = exporterProps.useGzip && rawPayload.size > exporterProps.compressionThreshold.toBytes()

            if (shouldCompress) {
                payload = compressGzip(rawPayload)
            }

            for (attempt in 1..maxAttempts) {
                if (sendRequest(payload, shouldCompress)) return

                val jitter = ThreadLocalRandom.current().nextDouble(exporterProps.jitterMin, exporterProps.jitterMax)
                val backoffDelay = (retryInterval * exporterProps.backoffMultiplier.pow((attempt - 1).toDouble()) * jitter).toLong()
                delay(backoffDelay.milliseconds)
            }
        } catch (e: Exception) {
            log.error(e) { "Critical error during Trace batch encoding/sending" }
        } finally {
            recycleBatch(events)
        }
    }

    private suspend fun sendRequest(payload: ByteArray, isCompressed: Boolean): Boolean {
        val builder = HttpRequest.newBuilder()
            .uri(exporterProps.url)
            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PROTOBUF_VALUE)
            .timeout(exporterProps.requestTimeout)
            .POST(HttpRequest.BodyPublishers.ofByteArray(payload))

        if (isCompressed) {
            builder.header(CONTENT_ENCODING, ENCODING_GZIP)
        }

        return try {
            val rs = httpClient.sendAsync(builder.build(), HttpResponse.BodyHandlers.discarding())
                .await()
            rs.statusCode() in 200..299
        } catch (ex: Exception) {
            log.warn { "Trace batch export failed due to network error: ${ex.message}" }
            false
        }
    }

    private fun compressGzip(data: ByteArray): ByteArray {
        val baos = ByteArrayOutputStream(data.size / 2)
        GZIPOutputStream(baos).use { it.write(data) }
        return baos.toByteArray()
    }

    private fun recycleBatch(batch: List<TraceEvent>) {
        val size = batch.size
        for (i in 0 until size) {
            batch[i].clearReferences()
            eventPool.offer(batch[i])
        }
    }

    companion object {
        private const val ENCODING_GZIP = "gzip"
    }
}
