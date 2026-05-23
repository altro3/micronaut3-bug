package com.micronaut.bug.trace.jdbc

object JdbcTraceKeywords {

    const val OP_SELECT = "SELECT"
    const val OP_INSERT = "INSERT"
    const val OP_UPDATE = "UPDATE"
    const val OP_DELETE = "DELETE"
    const val OP_DEFAULT = "SQL"

    @JvmStatic
    fun match(sql: String, start: Int, len: Int, keyword: String): Boolean {
        if (len != keyword.length) return false
        return sql.regionMatches(start, keyword, 0, len, ignoreCase = true)
    }

    @JvmStatic
    fun parseOperation(sql: String): String {
        var start = 0
        val len = sql.length

        while (start < len && sql[start] <= ' ') {
            start++
        }

        var end = start
        while (end < len && sql[end] > ' ') {
            end++
        }

        val wordLen = end - start
        if (wordLen <= 0) return OP_DEFAULT

        return when {
            match(sql, start, wordLen, OP_SELECT) -> OP_SELECT
            match(sql, start, wordLen, OP_INSERT) -> OP_INSERT
            match(sql, start, wordLen, OP_UPDATE) -> OP_UPDATE
            match(sql, start, wordLen, OP_DELETE) -> OP_DELETE
            else -> sql.substring(start, end).uppercase()
        }
    }
}
