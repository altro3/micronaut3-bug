package com.micronaut.bug.log

import ch.qos.logback.classic.pattern.ClassicConverter
import ch.qos.logback.classic.spi.ILoggingEvent

class MdcConverter : ClassicConverter() {

    override fun convert(event: ILoggingEvent): String {
        val mdc = event.mdcPropertyMap
        if (mdc.isEmpty()) {
            return EMPTY_RESULT
        }

        val customKeys = keys // Читаем volatile один раз
        val traceId = mdc[MDC_TRACE_ID]
        val targetId = mdc[MDC_TARGET_ID]

        // Если все основные и кастомные ключи пусты — выходим
        if (traceId == null && targetId == null && customKeys.isEmpty()) {
            return EMPTY_RESULT
        }

        val sb = threadLocalStringBuilder.get()
        sb.setLength(0)
        var hasContent = false

        // 1. Вывод traceId
        if (traceId != null) {
            sb.append(PREFIX).append(traceId)
            hasContent = true
        }

        // 2. Вывод targetId
        if (targetId != null) {
            if (hasContent) sb.append(SEPARATOR) else sb.append(PREFIX)
            sb.append(LABEL_TARGET).append(ASSIGN).append(targetId)
            hasContent = true
        }

        val size = customKeys.size
        for (i in 0 until size) {
            val key = customKeys[i]
            // Пропускаем уже обработанные системные ключи
            if (key == MDC_TRACE_ID || key == MDC_TARGET_ID) {
                continue
            }

            val value = mdc[key]
            if (value != null) {
                if (hasContent) sb.append(SEPARATOR) else sb.append(PREFIX)
                sb.append(key).append(ASSIGN).append(value)
                hasContent = true
            }
        }

        return if (hasContent) sb.append(POSTFIX).toString() else EMPTY_RESULT
    }

    companion object {
        const val MDC_TARGET_ID = "targetId"
        const val MDC_TRACE_ID = "traceId"

        @Volatile
        var keys: List<String> = emptyList()

        private const val LABEL_TARGET = "targetId"
        private const val ASSIGN = "="
        private const val SEPARATOR = ", "
        private const val PREFIX = "["
        private const val POSTFIX = "] "
        private const val EMPTY_RESULT = ""
        private const val ESTIMATED_SIZE = 128

        private val threadLocalStringBuilder = object : ThreadLocal<StringBuilder>() {
            override fun initialValue() = StringBuilder(ESTIMATED_SIZE)
        }
    }
}
