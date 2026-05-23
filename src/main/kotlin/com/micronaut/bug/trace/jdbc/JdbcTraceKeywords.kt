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
    fun parseOperationAndCollection(sql: String): Pair<String, String?> {
        var start = 0
        val len = sql.length

        while (start < len && sql[start] <= ' ') start++
        var end = start
        while (end < len && sql[end] > ' ') end++

        val wordLen = end - start
        if (wordLen <= 0) return Pair(OP_DEFAULT, null)

        return when {
            match(sql, start, wordLen, OP_SELECT) -> Pair(OP_SELECT, extractTableAfterKeyword(sql, end, "FROM"))
            match(sql, start, wordLen, OP_INSERT) -> Pair(OP_INSERT, extractTableAfterKeyword(sql, end, "INTO"))
            match(sql, start, wordLen, OP_UPDATE) -> Pair(OP_UPDATE, extractFirstWord(sql, end))
            match(sql, start, wordLen, OP_DELETE) -> Pair(OP_DELETE, extractTableAfterKeyword(sql, end, "FROM"))
            else -> Pair(OP_DEFAULT, null)
        }
    }

    @JvmStatic
    private fun extractTableAfterKeyword(sql: String, fromIdx: Int, keyword: String): String? {
        val len = sql.length
        var currentIdx = fromIdx

        while (currentIdx < len) {
            val idx = sql.indexOf(keyword, currentIdx, ignoreCase = true)
            if (idx == -1) return null

            // Защита от ложных совпадений внутри слов (например, колонка с именем user_from_city)
            val before = sql[idx - 1]
            val afterIdx = idx + keyword.length
            val after = if (afterIdx < len) sql[afterIdx] else ' '

            if (before <= ' ' && after <= ' ') {
                return extractFirstWord(sql, afterIdx)
            }
            currentIdx = idx + 1
        }
        return null
    }

    @JvmStatic
    private fun extractFirstWord(sql: String, startIdx: Int): String? {
        var start = startIdx
        val len = sql.length

        // Пропускаем пробелы и начальные экранирующие кавычки
        while (start < len && (sql[start] <= ' ' || sql[start] == '"' || sql[start] == '`')) {
            start++
        }
        var end = start

        // Ищем конец имени таблицы, останавливаемся на пробелах, скобках, запятых или закрывающих кавычках
        while (end < len && sql[end] > ' ' && sql[end] != '(' && sql[end] != '"' && sql[end] != '`' && sql[end] != ',' && sql[end] != ';') {
            end++
        }
        if (start == end) return null

        // Единственная аллокация строки — имя коллекции/таблицы (например, "users")
        return sql.substring(start, end)
    }
}
