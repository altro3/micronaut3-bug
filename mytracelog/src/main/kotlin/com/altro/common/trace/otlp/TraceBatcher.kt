package com.altro.common.trace.otlp

import com.altro.common.trace.config.TraceProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import io.opentelemetry.proto.trace.v1.Span
import io.opentelemetry.proto.trace.v1.Status.StatusCode
import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.withTimeoutOrNull
import java.net.http.HttpClient
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.LongAdder
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes

class TraceBatcher(
    appName: String,
    nodeName: String,
    traceProps: TraceProperties,
    httpClient: HttpClient
) {
    private val log = KotlinLogging.logger {}
    private val exporterProps = traceProps.export

    private val channel = Channel<TraceEvent>(exporterProps.queueCapacity)
    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val eventPool = ConcurrentLinkedQueue<TraceEvent>()
    private val sender = OtlpHttpSender(appName, nodeName, traceProps, httpClient, eventPool)

    private val concurrencySemaphore = Semaphore(exporterProps.maxSenders)

    private val droppedSpansCounter = LongAdder()

    // Четкое разделение джоб для контролируемого shutdown
    private var workerJob: Job? = null
    private var reporterJob: Job? = null

    fun enqueue(
        traceIdHex: String, spanIdHex: String, parentIdHex: String?, name: String,
        startEpochNanos: Long, endEpochNanos: Long, status: StatusCode, kind: Span.SpanKind,
        userAttrs: Map<String, Any>?, baggage: Map<String, String>?, propagationHeaders: Map<String, String>?,
        error: Throwable?,
    ) {
        if (!exporterProps.enabled) return

        val event = eventPool.poll() ?: TraceEvent()
        event.update(
            traceIdHex, spanIdHex, parentIdHex, name, startEpochNanos,
            endEpochNanos, status, kind, userAttrs, baggage, propagationHeaders, error
        )

        val result = channel.trySend(event)
        if (result.isFailure) {
            droppedSpansCounter.increment()
            event.clearReferences()
            eventPool.offer(event)
        }
    }

    @PostConstruct
    fun start() {
        if (!exporterProps.enabled) return

        workerJob = exportScope.launch {
            log.info { "Starting Trace batcher worker..." }
            var batch = ArrayList<TraceEvent>(exporterProps.batchSize)
            val flushIntervalMs = exporterProps.flushInterval.toMillis()

            while (isActive) {
                try {
                    val result = channel.receiveCatching()
                    if (result.isClosed) {
                        flushRemainingOnShutdown(batch)
                        break
                    }

                    batch.add(result.getOrThrow())
                    val startTime = System.currentTimeMillis()

                    while (batch.size < exporterProps.batchSize) {
                        val nextResult = channel.tryReceive()
                        if (nextResult.isSuccess) {
                            val event = nextResult.getOrNull()
                            if (event != null) batch.add(event)
                        } else {
                            if (nextResult.isClosed) break

                            val remainingTime = flushIntervalMs - (System.currentTimeMillis() - startTime)
                            if (remainingTime <= 0) break

                            delay(remainingTime.milliseconds)

                            var drainResult = channel.tryReceive()
                            while (drainResult.isSuccess && batch.size < exporterProps.batchSize) {
                                val ev = drainResult.getOrNull()
                                if (ev != null) batch.add(ev)
                                drainResult = channel.tryReceive()
                            }
                            break
                        }
                    }

                    if (batch.isNotEmpty()) {
                        dispatchBatch(batch)
                        batch = ArrayList(exporterProps.batchSize)
                    }

                } catch (_: ClosedReceiveChannelException) {
                    flushRemainingOnShutdown(batch)
                    break
                } catch (_: CancellationException) {
                    break
                } catch (e: Exception) {
                    log.error(e) { "Error in Trace batcher loop." }
                    delay(exporterProps.retryInterval.toMillis().milliseconds)
                }
            }
        }

        reporterJob = exportScope.launch {
            while (isActive) {
                try {
                    delay(1.minutes)
                    val droppedCount = droppedSpansCounter.sumThenReset()
                    if (droppedCount > 0) {
                        log.warn { "Trace queue overflow detected. Dropped $droppedCount spans in the last minute. Consider increasing queueCapacity or batchSize." }
                    }
                } catch (_: CancellationException) {
                    break
                } catch (e: Exception) {
                    log.error(e) { "Error in Trace dropped spans reporter loop." }
                }
            }
        }
    }

    private suspend fun dispatchBatch(batch: ArrayList<TraceEvent>) {
        concurrencySemaphore.acquire()

        exportScope.launch {
            try {
                sender.sendBatch(batch)
            } finally {
                concurrencySemaphore.release()
            }
        }
    }

    @PreDestroy
    fun stop() {
        if (!exporterProps.enabled) return
        log.info { "Shutting down Trace batcher..." }

        reporterJob?.cancel()
        channel.close()

        runBlocking {
            withTimeoutOrNull(exporterProps.shutdownTimeout.toMillis().milliseconds) {
                workerJob?.join()
                true
            }
        }
        exportScope.cancel()
    }

    private suspend fun flushRemainingOnShutdown(batch: ArrayList<TraceEvent>) {
        var result = channel.tryReceive()
        while (result.isSuccess) {
            val next = result.getOrNull()
            if (next != null) {
                batch.add(next)
                if (batch.size >= exporterProps.batchSize) {
                    sender.sendBatch(batch)
                    batch.clear()
                }
            }
            result = channel.tryReceive()
        }
        if (batch.isNotEmpty()) {
            sender.sendBatch(batch)
            batch.clear()
        }
    }
}
