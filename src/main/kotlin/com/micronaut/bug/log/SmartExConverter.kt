package com.micronaut.bug.log

import org.apache.logging.log4j.core.LogEvent
import org.apache.logging.log4j.core.config.plugins.Plugin
import org.apache.logging.log4j.core.pattern.ConverterKeys
import org.apache.logging.log4j.core.pattern.LogEventPatternConverter

@Plugin(name = "SmartExConverter", category = "Converter")
@ConverterKeys("uEx")
class SmartExConverter : LogEventPatternConverter("SmartEx", "uEx") {
    override fun format(event: LogEvent, toAppendTo: StringBuilder) {
        val proxy = event.thrownProxy ?: return
        LogUtil.formatStackTrace(
            proxy = proxy,
            sb = toAppendTo,
            maxLines = maxLines,
            rootCauseFull = rootCauseFull,
        )
    }

    override fun handlesThrowable(): Boolean = true

    companion object {
        @Volatile
        var maxLines: Int = 10

        @Volatile
        var rootCauseFull: Boolean = true

        @JvmStatic
        fun newInstance(options: Array<String>?): SmartExConverter = SmartExConverter()
    }
}
