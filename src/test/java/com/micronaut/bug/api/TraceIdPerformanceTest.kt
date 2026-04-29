package com.micronaut.bug.api

import com.micronaut.bug.trace.TraceIdGenerator
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import java.util.UUID
import kotlin.system.measureNanoTime

class TraceIdPerformanceTest {

    private val iterations = 1_000_000

    @Test
    @DisplayName("Проверка корректности формата")
    fun testFormat() {
        val traceId = TraceIdGenerator.generate()
        assertEquals(32, traceId.length, "Длина должна быть 32 символа")
        assertTrue(traceId.matches(Regex("^[0-9a-f]{32}$")), "Должен быть валидный hex")
    }

    @Test
    @DisplayName("Сравнение производительности: Standard UUID vs TraceIdGenerator")
    fun comparePerformance() {
        // 1. Прогрев (Warmup) - критично для JIT
        repeat(100_000) {
            UUID.randomUUID().toString().replace("-", "")
            TraceIdGenerator.generate()
        }

        // 2. Тест стандартного метода
        var sink = 0 // Чтобы JVM не удалила цикл
        val timeUuid = measureNanoTime {
            repeat(iterations) {
                val s = UUID.randomUUID().toString().replace("-", "")
                sink += s.length
            }
        }

        // 3. Тест оптимизированного метода
        val timeFast = measureNanoTime {
            repeat(iterations) {
                val s = TraceIdGenerator.generate()
                sink += s.length
            }
        }

        println("--- Результаты для $iterations итераций ---")
        println("Standard UUID.replace: $timeUuid ms")
        println("TraceIdGenerator:      $timeFast ms")

        val ratio = timeUuid.toDouble() / timeFast
        println("Выигрыш в скорости:    ${"%.2f".format(ratio)}x")

        assertTrue(timeFast < timeUuid, "Наш генератор должен быть быстрее")
    }
}
