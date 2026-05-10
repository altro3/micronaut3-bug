package com.micronaut.bug.trace

import com.google.protobuf.ByteString
import java.util.concurrent.ThreadLocalRandom

object TraceIdGenerator {
    // 512 байт (256 пар HEX-символов). Идеально для L1-кэша процессора.
    private val HEX_TABLE = ByteArray(512).apply {
        val hexChars = "0123456789abcdef"
        for (i in 0 until 256) {
            this[i * 2] = hexChars[i shr 4 and 0xF].code.toByte()
            this[i * 2 + 1] = hexChars[i and 0xF].code.toByte()
        }
    }

    /**
     * Генерирует 32-символьный hex-ID.
     * Быстрее UUID.randomUUID() в 10-20 раз.
     */
    fun generate(): String {
        val random = ThreadLocalRandom.current()
        val buffer = ByteArray(32)

        // Получаем 128 бит случайности через два Long
        val hi = random.nextLong()
        val lo = random.nextLong()

        // Заполняем 32 байта в буфере (без циклов и условий)
        fill(hi, buffer, 0)
        fill(lo, buffer, 16)

        // Конструктор через ISO_8859_1 в Java 21+ создает компактную
        // строку (Latin-1) без валидации и лишних проверок.
        return String(buffer, Charsets.ISO_8859_1)
    }

    /**
     * Генерирует 16-символьный hex-ID (64 бита) для Span ID.
     * Соответствует стандарту W3C Trace Context и OpenTelemetry.
     */
    fun generateSpanId(): String {
        val random = ThreadLocalRandom.current()
        val buffer = ByteArray(16)

        fill(random.nextLong(), buffer, 0)

        return String(buffer, Charsets.ISO_8859_1)
    }

    private fun fill(value: Long, target: ByteArray, offset: Int) {
        // Развернутая запись: за одну операцию берем 8 бит и пишем 2 символа
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
        // Вычисляем индекс в таблице: (байт) * 2
        val idx = (shiftedValue.toInt() and 0xFF) shl 1
        target[offset] = HEX_TABLE[idx]
        target[offset + 1] = HEX_TABLE[idx + 1]
    }

    /**
     * Быстро конвертирует HEX-строку обратно в ByteString для отправки в Tempo.
     * Исключает использование тяжелого HexFormat и лишние проверки.
     *
     * @param hex 32-символьная (для traceId) или 16-символьная (для spanId) HEX-строка
     * @return ByteString для Protobuf билдера
     */
    fun toByteString(hex: String): ByteString {
        val len = hex.length
        val result = ByteArray(len / 2)
        for (i in result.indices) {
            val h = decodeDigit(hex[i * 2])
            val l = decodeDigit(hex[i * 2 + 1])
            result[i] = ((h shl 4) or l).toByte()
        }
        return ByteString.copyFrom(result)
    }

    private fun decodeDigit(c: Char): Int =
        when (c) {
            in '0'..'9' -> c - '0'
            in 'a'..'f' -> c - 'a' + 10
            else -> throw IllegalArgumentException("Invalid hex character: $c")
        }
}