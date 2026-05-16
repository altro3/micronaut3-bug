package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.core.io.Resource
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.init.ScriptUtils
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional

open class FlywayRollbackStepExecutor(
    private val jdbcTemplate: JdbcTemplate
) {
    private val log = KotlinLogging.logger {}

    @Transactional(propagation = Propagation.REQUIRES_NEW, rollbackFor = [Exception::class])
    open fun executeRollbackStep(
        version: String,
        scriptResource: Resource?,
        isNoRollbackMigration: Boolean,
        activeSchema: String,
        historyTable: String
    ) {
        log.info { "Processing atomic rollback step for version: $version" }

        // Гарантируем правильную схему (совместимо с Postgres)
        if (activeSchema.isNotBlank() && activeSchema != "public") {
            jdbcTemplate.execute("SET search_path TO $activeSchema")
        }

        if (isNoRollbackMigration) {
            log.info { "Migration $version is explicitly marked as non-rollable. Skipping SQL execution, updating history only." }
        } else {
            val resource = scriptResource
                ?: throw IllegalStateException("Rollback failed! No undo script resource provided for version $version")

            log.info { "Found native-style undo script: ${resource.filename}. Executing SQL..." }

            val connection = jdbcTemplate.dataSource?.connection
                ?: throw IllegalStateException("Failed to obtain database connection from DataSource")

            connection.use { conn ->
                // Исправлено: Явно выставляем схему для текущего коннекта БД
                if (activeSchema.isNotBlank()) {
                    try {
                        // Универсальный способ для большинства СУБД
                        conn.schema = activeSchema
                    } catch (_: Exception) {
                        // Фолбэк для старых драйверов Postgres / H2
                        conn.createStatement().use { stmt ->
                            stmt.execute("SET search_path TO $activeSchema")
                        }
                    }
                    log.debug { "Database connection schema set to '$activeSchema'" }
                }

                // Теперь скрипт выполнится строго в контексте нужной схемы!
                ScriptUtils.executeSqlScript(conn, resource)
            }
        }

        // Удаляем запись о миграции из таблицы истории Flyway
        val deletedRows = jdbcTemplate.update(
            "DELETE FROM $historyTable WHERE version = ?",
            version
        )

        if (deletedRows == 0) {
            throw IllegalStateException("SQL executed, but failed to remove version $version from metadata table $historyTable")
        }

        log.info { "Metadata successfully synchronized. Version $version is now marked as unapplied." }
    }
}
