package com.micronaut.bug.log

import org.apache.logging.log4j.core.impl.ThrowableProxy

object LogUtil {

    fun formatStackTrace(
        proxy: ThrowableProxy,
        sb: StringBuilder,
        maxLines: Int = 10,
        rootCauseFull: Boolean = true
    ): String {

        // 1. Первое исключение
        renderProxy(proxy, sb, maxLines, isRoot = false)

        // 2. Идем по цепочке Cause
        var cause = proxy.causeProxy
        while (cause != null) {
            if (cause.causeProxy == null) {
                // Последний Cause
                sb.append("Root Cause: ")
                val limit = if (rootCauseFull) Int.MAX_VALUE else maxLines
                renderProxy(cause, sb, limit, isRoot = true)
            } else {
                // Промежуточный Cause - только заголовок (согласно твоей идее)
                sb.append("Caused by: ").append(cause.name).append(": ")
                    .append(cause.message).append(" [intermediate skipped]\n")
            }
            cause = cause.causeProxy
        }

        return sb.toString()
    }

    private fun renderProxy(proxy: ThrowableProxy, sb: StringBuilder, limit: Int, isRoot: Boolean) {
        sb.append(proxy.name).append(": ").append(proxy.message).append("\n")

        val extendedStackTrace = proxy.extendedStackTrace
        val linesToShow = if (isRoot) extendedStackTrace.size else minOf(extendedStackTrace.size, limit)

        for (i in 0 until linesToShow) {
            sb.append("\tat ").append(extendedStackTrace[i]).append("\n")
        }

        if (extendedStackTrace.size > linesToShow) {
            val remaining = extendedStackTrace.size - linesToShow
            sb.append("\t... and ").append(remaining).append(" more lines\n")
        }
    }
}
