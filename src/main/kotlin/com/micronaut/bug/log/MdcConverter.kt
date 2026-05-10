package com.micronaut.bug.log

import org.apache.logging.log4j.core.LogEvent
import org.apache.logging.log4j.core.config.plugins.Plugin
import org.apache.logging.log4j.core.pattern.ConverterKeys
import org.apache.logging.log4j.core.pattern.LogEventPatternConverter
import org.apache.logging.log4j.util.ReadOnlyStringMap

@Plugin(name = "MdcConverter", category = "Converter")
@ConverterKeys("uMDC")
class MdcConverter : LogEventPatternConverter("MdcConverter", "mdc") {

    override fun format(event: LogEvent, toAppendTo: StringBuilder) {
        val mdc: ReadOnlyStringMap = event.contextData
        val customKeys = keys // Читаем volatile один раз

        if (mdc.isEmpty) return

        val traceId = mdc.getValue<String>(MDC_TRACE_ID)
        val targetId = mdc.getValue<String>(MDC_TARGET_ID)

        if (traceId == null && targetId == null && customKeys.isEmpty()) {
            return
        }

        var hasContent = false

        // 1. Вывод traceId
        if (traceId != null) {
            toAppendTo.append(PREFIX).append(traceId)
            hasContent = true
        }

        // 2. Вывод targetId
        if (targetId != null) {
            if (hasContent) toAppendTo.append(SEPARATOR) else toAppendTo.append(PREFIX)
            toAppendTo.append(LABEL_TARGET).append(ASSIGN).append(targetId)
            hasContent = true
        }

        // 3. Вывод кастомных ключей
        for (i in customKeys.indices) {
            val key = customKeys[i]
            if (key == MDC_TRACE_ID || key == MDC_TARGET_ID) continue

            val value = mdc.getValue<String>(key)
            if (value != null) {
                if (hasContent) toAppendTo.append(SEPARATOR) else toAppendTo.append(PREFIX)
                toAppendTo.append(key).append(ASSIGN).append(value)
                hasContent = true
            }
        }

        if (hasContent) {
            toAppendTo.append(POSTFIX).append(SPACE)
        }
    }

    companion object {
        const val MDC_TARGET_ID = "targetId"
        const val MDC_TRACE_ID = "traceId"

        @Volatile
        var keys: List<String> = emptyList()

        private const val LABEL_TARGET = "targetId"
        private const val ASSIGN = '='
        private const val SEPARATOR = ", "
        private const val PREFIX = '['
        private const val POSTFIX = ']'
        private const val SPACE = ' '

        @Suppress("unused")
        @JvmStatic
        fun newInstance(options: Array<String>?): MdcConverter = MdcConverter()
    }
}
