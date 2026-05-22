package com.micronaut.bug.trace.otlp

import com.micronaut.bug.trace.config.TraceProperties
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
import kotlin.time.Duration.Companion.milliseconds

class TraceBatcher(
    appName: String,
    nodeName: String,
    traceProps: TraceProperties,
    httpClient: HttpClient
) {
    private val log = KotlinLogging.logger {}
    private val exporterProps = traceProps.exporter

    private val channel = Channel<TraceEvent>(exporterProps.queueCapacity)
    private val exportScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val eventPool = ConcurrentLinkedQueue<TraceEvent>()

    private val sender = OtlpHttpSender(appName, nodeName, traceProps, httpClient, eventPool)

    // БАГФИКС: Корутинный неблокирующий семафор вместо java.util.concurrent.Semaphore.
    // Он не блокирует поток ядра ОС, а мягко приостанавливает корутину главного воркера.
    private val concurrencySemaphore = Semaphore(exporterProps.maxSenders)

    fun enqueue(
        traceIdHex: String,
        spanIdHex: String,
        parentIdHex: String?,
        name: String,
        startEpochNanos: Long,
        endEpochNanos: Long,
        status: StatusCode,
        kind: Span.SpanKind,
        userAttrs: Map<String, Any>?,
        baggage: Map<String, String>?,
        propagationHeaders: Map<String, String>?,
        error: Throwable? = null,
    ) {
        if (!exporterProps.enabled) return

        val event = eventPool.poll() ?: TraceEvent()
        event.update(
            traceIdHex, spanIdHex, parentIdHex, name, startEpochNanos,
            endEpochNanos, status, kind, userAttrs, baggage, propagationHeaders, error
        )

        val result = channel.trySend(event)
        if (result.isFailure) {
            log.warn { "Trace queue overflow, span dropped: traceId=$traceIdHex" }
            event.clearReferences()
            eventPool.offer(event)
        }
    }

    @PostConstruct
    fun start() {
        if (!exporterProps.enabled) return

        exportScope.launch {
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

                            val nextResTimeout = withTimeoutOrNull(remainingTime.milliseconds) {
                                channel.receiveCatching()
                            }

                            if (nextResTimeout == null) {
                                break
                            } else if (nextResTimeout.isSuccess) {
                                batch.add(nextResTimeout.getOrThrow())
                            } else {
                                break
                            }
                        }
                    }

                    // ОПТИМИЗАЦИЯ ПАМЯТИ: Ждем свободный сетевой слот ДО того, как породить новый ArrayList.
                    // Если сеть тормозит, главный воркер заснет тут, канал channel начнет заполняться,
                    // и лишние трейсы безопасно сбросятся в enqueue через trySend. Мусор в хипе расти не будет!
                    concurrencySemaphore.acquire()

                    dispatchBatch(batch)
                    batch = ArrayList(exporterProps.batchSize)

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
    }

    private fun dispatchBatch(batch: ArrayList<TraceEvent>) {
        if (batch.isEmpty()) return

        exportScope.launch {
            try {
                sender.sendBatch(batch)
            } finally {
                // Освобождаем слот строго после окончания отправки
                concurrencySemaphore.release()
            }
        }
    }

    @PreDestroy
    fun stop() {
        if (!exporterProps.enabled) return
        log.info { "Shutting down Trace batcher..." }
        channel.close()

        runBlocking {
            val job = exportScope.coroutineContext[Job]
            withTimeoutOrNull(exporterProps.shutdownTimeout.toMillis().milliseconds) {
                // Даем воркеру корректно обработать flushRemainingOnShutdown
                job?.children?.forEach { it.join() }
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
                    sender.sendBatch(batch) // При шатдауне шлем синхронно, sender сам вернет в пул
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
