package com.micronaut.bug.api

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Fork
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.TearDown
import org.openjdk.jmh.annotations.Warmup
import org.slf4j.LoggerFactory
import org.slf4j.MDC
import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput) // Меряем количество операций в секунду
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 3, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 5, timeUnit = TimeUnit.SECONDS)
@Fork(1)
open class LoggingBenchmark {

    // Используем логгер через SLF4J фасад
    private val logger = LoggerFactory.getLogger(LoggingBenchmark::class.java)

    @Setup
    fun setup() {
        // Заполняем MDC, так как в реальном приложении он всегда есть (traceId, userId)
        MDC.put("traceId", "test-trace-id-12345")
        MDC.put("userId", "user-999")
    }

    @Benchmark
    fun testLogThroughput() {
        // Типичный бизнес-лог с параметрами
        logger.info("Order {} processed for customer {} in {} ms", 1001, "AcmeCorp", 42)
    }

    @TearDown
    fun tearDown() {
        MDC.clear()
    }
}