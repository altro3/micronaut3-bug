package com.altro.common.trace.jdbc

object JdbcUrlParser {

    data class ConnectionInfo(val host: String, val port: Long, val namespace: String?)

    // 1. Вынесли дефолтное значение в константу для предотвращения лишних аллокаций
    val DEFAULT_CONNECTION_INFO = ConnectionInfo("localhost", 5432L, null)

    @JvmStatic
    fun parse(jdbcUrl: String, defaultSystem: String): ConnectionInfo {
        try {
            if (jdbcUrl.isBlank() || !jdbcUrl.startsWith("jdbc:")) {
                return DEFAULT_CONNECTION_INFO
            }

            val cleanUrl = jdbcUrl.substring(5)
            val schemeEnd = cleanUrl.indexOf("://")
            if (schemeEnd == -1) return DEFAULT_CONNECTION_INFO

            val remaining = cleanUrl.substring(schemeEnd + 3)
            val authorityAndPath = remaining.substringBefore('?')

            val slashIdx = authorityAndPath.indexOf('/')
            val authority = if (slashIdx != -1) authorityAndPath.substring(0, slashIdx) else authorityAndPath
            val path = if (slashIdx != -1) authorityAndPath.substring(slashIdx + 1).takeIf { it.isNotBlank() } else null

            val colonIdx = authority.indexOf(':')
            val host = if (colonIdx != -1) authority.substring(0, colonIdx) else authority

            val defaultPort = when (defaultSystem.lowercase()) {
                "postgresql" -> 5432L
                "mysql", "mariadb" -> 3306L
                "oracle" -> 1521L
                "sqlserver" -> 1433L
                else -> 80L
            }

            val port = if (colonIdx != -1) {
                authority.substring(colonIdx + 1).toLongOrNull() ?: defaultPort
            } else defaultPort

            return ConnectionInfo(host, port, path)
        } catch (_: Throwable) {
            return DEFAULT_CONNECTION_INFO
        }
    }
}
