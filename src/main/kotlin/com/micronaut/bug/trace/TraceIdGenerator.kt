package com.micronaut.bug.trace

import java.util.concurrent.ThreadLocalRandom

object TraceIdGenerator {

    private val HEX_TABLE = ByteArray(512).apply {
        val hexChars = "0123456789abcdef"
        for (i in 0 until 256) {
            this[i * 2] = hexChars[i shr 4 and 0xF].code.toByte()
            this[i * 2 + 1] = hexChars[i and 0xF].code.toByte()
        }
    }

    fun generate(): String {
        val random = ThreadLocalRandom.current()
        val buffer = ByteArray(32)

        val hi = random.nextLong()
        val lo = random.nextLong()

        fill(hi, buffer, 0)
        fill(lo, buffer, 16)

        return String(buffer, Charsets.ISO_8859_1)
    }

    fun generateSpanId(): String {
        val random = ThreadLocalRandom.current()
        val buffer = ByteArray(16)

        fill(random.nextLong(), buffer, 0)

        return String(buffer, Charsets.ISO_8859_1)
    }

    private fun fill(value: Long, target: ByteArray, offset: Int) {
        writePair(target, offset + 0, value shr 56)
        writePair(target, offset + 2, value shr 48)
        writePair(target, offset + 4, value shr 40)
        writePair(target, offset + 6, value shr 32)
        writePair(target, offset + 8, value shr 24)
        writePair(target, offset + 10, value shr 16)
        writePair(target, offset + 12, value shr 8)
        writePair(target, offset + 14, value)
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun writePair(target: ByteArray, offset: Int, shiftedValue: Long) {
        val idx = (shiftedValue.toInt() and 0xFF) shl 1
        target[offset] = HEX_TABLE[idx]
        target[offset + 1] = HEX_TABLE[idx + 1]
    }
}