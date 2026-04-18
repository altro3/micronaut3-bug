package com.micronaut.bug.config.log

import ch.qos.logback.classic.pattern.ClassicConverter
import ch.qos.logback.classic.spi.ILoggingEvent

/**
 * Кастомный конвертер Logback для форматированного вывода MDC.
 * Выводит идентификаторы в строгом порядке: rqId, targetId, extRqId, затем остальные ключи.
 */
class MdcConverter : ClassicConverter() {

    override fun convert(event: ILoggingEvent): String {
        val mdc = event.mdcPropertyMap
        if (mdc.isEmpty()) {
            return EMPTY_RESULT
        }

        val customKeys = keys // Читаем volatile один раз
        val rqId = mdc[X_REQ_ID]
        val targetId = mdc[TARGET_ID]
        val extRqId = mdc[EXT_RQ_ID]

        // Если все основные и кастомные ключи пусты — выходим
        if (rqId == null && targetId == null && extRqId == null && customKeys.isEmpty()) {
            return EMPTY_RESULT
        }

        val sb = threadLocalStringBuilder.get()
        sb.setLength(0)
        var hasContent = false

        // 1. Вывод rqId
        if (rqId != null) {
            sb.append(PREFIX).append(LABEL_RQ).append(ASSIGN).append(rqId)
            hasContent = true
        }

        // 2. Вывод targetId
        if (targetId != null) {
            if (hasContent) sb.append(SEPARATOR) else sb.append(PREFIX)
            sb.append(LABEL_TARGET).append(ASSIGN).append(targetId)
            hasContent = true
        }

        // 3. Вывод extRqId (идентификатор внешнего вызова)
        if (extRqId != null) {
            if (hasContent) sb.append(SEPARATOR) else sb.append(PREFIX)
            sb.append(LABEL_EXT_RQ).append(ASSIGN).append(extRqId)
            hasContent = true
        }

        // 4. Вывод остальных ключей из конфигурации
        val size = customKeys.size
        for (i in 0 until size) {
            val key = customKeys[i]
            // Пропускаем уже обработанные системные ключи
            if (key == X_REQ_ID || key == TARGET_ID || key == EXT_RQ_ID) {
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
        const val X_REQ_ID = "x-req-id"
        const val TARGET_ID = "targetId"
        const val EXT_RQ_ID = "extRqId"

        @Volatile
        var keys: List<String> = emptyList()

        private const val LABEL_RQ = "rqId"
        private const val LABEL_TARGET = "targetId"
        private const val LABEL_EXT_RQ = "extRqId"

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
