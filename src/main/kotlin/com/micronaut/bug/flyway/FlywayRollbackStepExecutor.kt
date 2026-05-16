package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.io.Resource
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.init.ScriptUtils

open class FlywayRollbackStepExecutor(
    private val jdbcTemplate: JdbcTemplate
) {
    private val log = KotlinLogging.logger {}

    open fun executeRollbackStep(
        version: String,
        scriptResource: Resource?,
        isNoRollbackMigration: Boolean,
        activeSchema: String,
        historyTable: String
    ) {
        log.info { "Processing atomic rollback step for version: $version" }

        val dataSource = jdbcTemplate.dataSource
            ?: throw IllegalStateException("Failed to obtain database connection from DataSource")

        val fullTableName = if (activeSchema.isNotBlank()) "$activeSchema.$historyTable" else historyTable

        // Берем физическое соединение из пула
        val conn = dataSource.connection
        try {
            // Включаем ручное управление транзакцией (выключаем auto-commit)
            conn.autoCommit = false

            // Выставляем схему для текущего соединения
            if (activeSchema.isNotBlank()) {
                try {
                    conn.schema = activeSchema
                } catch (_: Exception) {
                    conn.createStatement().use { stmt ->
                        stmt.execute("SET search_path TO $activeSchema")
                    }
                }
                log.debug { "Database connection schema set to '$activeSchema'" }
            }

            if (isNoRollbackMigration) {
                log.info { "Migration $version is explicitly marked as non-rollable. Skipping SQL execution, updating history only." }
            } else {
                val resource = scriptResource
                    ?: throw IllegalStateException("Rollback failed! No undo script resource provided for version $version")

                log.info { "Found native-style undo script: ${resource.filename}. Executing SQL..." }

                // Выполняем скрипт отката строго в контексте нашей ручной транзакции
                ScriptUtils.executeSqlScript(conn, resource)
            }

            // Удаляем запись о миграции из таблицы истории Flyway через то же самое соединение!
            conn.prepareStatement("DELETE FROM $fullTableName WHERE version = ?").use { stmt ->
                stmt.setString(1, version)
                val deletedRows = stmt.executeUpdate()

                if (deletedRows == 0) {
                    throw IllegalStateException("SQL executed, but failed to remove version $version from metadata table $fullTableName")
                }
            }

            // Если всё прошло успешно и без ошибок — фиксируем транзакцию в БД
            conn.commit()
            log.info { "Metadata successfully synchronized. Version $version is now marked as unapplied." }

        } catch (e: Exception) {
            // В случае любой ошибки (ошибка в SQL или сбой удаления) откатываем всю транзакцию назад
            log.error(e) { "Failed to execute rollback step for version $version. Rolling back transaction." }
            try {
                conn.rollback()
            } catch (rollbackEx: Exception) {
                log.error(rollbackEx) { "Failed to rollback transaction after error" }
            }
            throw e // Пробрасываем ошибку выше, чтобы сработал Fail-Fast в Engine
        } finally {
            // Гарантированно закрываем соединение и возвращаем его в пул ресурсов
            conn.close()
        }
    }
}
