package com.altro.common.log

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
        val userId = mdc.getValue<String>(MDC_USER_ID)

        if (traceId == null && userId == null && customKeys.isEmpty()) {
            return
        }

        var hasContent = false

        if (traceId != null) {
            toAppendTo.append(PREFIX).append(traceId)
            hasContent = true
        }

        if (userId != null) {
            if (hasContent) toAppendTo.append(SEPARATOR) else toAppendTo.append(PREFIX)
            toAppendTo.append(userId)
            hasContent = true
        }

        for (i in customKeys.indices) {
            val key = customKeys[i]
            if (key == MDC_TRACE_ID || key == MDC_USER_ID) continue

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
        const val MDC_USER_ID = "userId"
        const val MDC_TRACE_ID = "traceId"

        @Volatile
        var keys: List<String> = emptyList()

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
