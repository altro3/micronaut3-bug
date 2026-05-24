package com.altro.common.log

import org.apache.logging.log4j.core.impl.ThrowableProxy

object LogUtil {

    private const val SUFFIX_TRUNCATED = "... [TRUNCATED]"

    fun formatStackTrace(t: Throwable, sb: StringBuilder, maxLines: Int, rootCauseFull: Boolean = true): String {
        sb.setLength(0)
        var current: Throwable? = t
        while (current != null) {
            val isRoot = current.cause == null
            val isFirst = current === t

            if (isFirst || isRoot) {
                if (isRoot && !isFirst) sb.append("Root Cause: ")
                renderBlock(sb, maxLines, isRoot && rootCauseFull, current.javaClass.name, current.message, current.stackTrace)
            } else {
                sb.append("Caused by: ").append(current.javaClass.name).append(": ")
                    .append(current.message ?: "").append(" [intermediate skipped]\n")
            }
            current = current.cause
        }
        return sb.toString()
    }

    fun formatStackTrace(proxy: ThrowableProxy, sb: StringBuilder, maxLines: Int, rootCauseFull: Boolean = true): String {
        var current: ThrowableProxy? = proxy
        while (current != null) {
            val isRoot = current.causeProxy == null
            val isFirst = current === proxy

            if (isFirst || isRoot) {
                if (isRoot && !isFirst) sb.append("Root Cause: ")
                renderBlock(sb, maxLines, isRoot && rootCauseFull, current.name, current.message, current.extendedStackTrace)
            } else {
                sb.append("Caused by: ").append(current.name).append(": ")
                    .append(current.message ?: "").append(" [intermediate skipped]\n")
            }
            current = current.causeProxy
        }
        return sb.toString()
    }

    private fun renderBlock(sb: StringBuilder, limit: Int, isFull: Boolean, name: String, message: String?, stack: Array<*>) {
        sb.append(name).append(": ").append(message ?: "").append("\n")

        val linesToShow = if (isFull) stack.size else minOf(stack.size, limit)

        for (i in 0 until linesToShow) {
            sb.append("\tat ").append(stack[i]).append("\n")
        }

        if (stack.size > linesToShow) {
            sb.append("\t... and ").append(stack.size - linesToShow).append(" more lines\n")
        }
    }

    fun truncateIfNeeded(text: String, limitLogSize: Int, truncateChunkSize: Int): String {
        if (limitLogSize > 0 && text.length > limitLogSize) {
            val head = text.take(truncateChunkSize)
            val tail = text.takeLast(truncateChunkSize)
            return "$head\n$SUFFIX_TRUNCATED [Skipped ${text.length - 2 * truncateChunkSize} chars]\n$tail"
        }
        return text
    }
}
