package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.configuration.ClassicConfiguration
import org.flywaydb.core.internal.parser.ParsingContext
import org.flywaydb.core.internal.resource.StringResource
import org.flywaydb.database.postgresql.PostgreSQLParser
import org.springframework.core.io.Resource
import org.springframework.jdbc.core.JdbcTemplate
import java.sql.Connection
import java.sql.SQLException

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

        val escapedSchema = activeSchema.trim()
        val escapedTable = historyTable.trim()
        val fullTableName = if (escapedSchema.isNotBlank()) "\"$escapedSchema\".\"$escapedTable\"" else "\"$escapedTable\""

        val conn: Connection = try {
            dataSource.connection
        } catch (e: SQLException) {
            throw IllegalStateException("Failed to open connection from DataSource", e)
        }

        try {
            // =================================================================
            // ЭТАП 1: Выполнение SQL-скрипта отката (Первая транзакция)
            // =================================================================
            conn.autoCommit = true

            if (escapedSchema.isNotBlank()) {
                setSchema(conn, escapedSchema)
            }

            if (isNoRollbackMigration) {
                log.info { "Migration $version is explicitly marked as non-rollable. Skipping SQL execution." }
            } else {
                val resource = scriptResource
                    ?: throw IllegalStateException("Rollback failed! No undo script resource provided for version $version")

                log.info { "Found native-style undo script: ${resource.filename}. Parsing and executing..." }

                val sqlContent = resource.getContentAsString(Charsets.UTF_8)

                // Используем оригинальный парсер Flyway для PostgreSQL.
                // Он идеально понимает блоки $$, комментарии и разделители.
                val parser = PostgreSQLParser(ClassicConfiguration(), ParsingContext())
                val statementReader = parser.parse(StringResource(sqlContent))

                while (statementReader.hasNext()) {
                    val statement = statementReader.next() ?: continue
                    val sqlToExecute = statement.sql

                    if (sqlToExecute.isNullOrBlank()) {
                        continue
                    }

                    if (statement.canExecuteInTransaction() && conn.autoCommit) {
                        conn.autoCommit = false // Включаем транзакцию для обычных DDL/DML
                    } else if (!statement.canExecuteInTransaction() && !conn.autoCommit) {
                        conn.commit() // Фиксируем накопленное, если встретили нетранзакционную команду
                        conn.autoCommit = true
                    }

                    log.debug { "Executing statement: ${sqlToExecute.take(100).replace("\n", " ")}..." }
                    conn.createStatement().use { stmt ->
                        stmt.execute(sqlToExecute)
                    }
                }
            }

            // Если мы вышли из цикла и оставались в транзакции — фиксируем её
            if (!conn.autoCommit) {
                conn.commit()
            }
            log.debug { "Transaction for SQL script of version $version successfully committed." }

            // =================================================================
            // ЭТАП 2: Синхронизация таблицы истории Flyway (Вторая транзакция)
            // =================================================================
            conn.autoCommit = false // Явно открываем чистую транзакцию для метаданных
            conn.prepareStatement("DELETE FROM $fullTableName WHERE version = ?").use { stmt ->
                stmt.setString(1, version)
                val deletedRows = stmt.executeUpdate()

                if (deletedRows == 0) {
                    throw IllegalStateException("SQL executed, but failed to remove version $version from metadata table $fullTableName")
                }
            }

            // Фиксируем транзакцию метаданных
            conn.commit()
            log.info { "Metadata successfully synchronized. Version $version is now marked as unapplied." }

        } catch (e: Exception) {
            log.error(e) { "Failed to execute rollback step for version $version. Rolling back active transaction." }
            try {
                if (!conn.isClosed) {
                    conn.rollback()
                }
            } catch (rollbackEx: Exception) {
                log.error(rollbackEx) { "Failed to rollback transaction after error" }
            }
            throw e
        } finally {
            try {
                conn.close()
            } catch (closeEx: Exception) {
                log.error(closeEx) { "Failed to close database connection" }
            }
        }
    }

    private fun setSchema(conn: Connection, schema: String) {
        try {
            conn.schema = schema
        } catch (_: Exception) {
            conn.createStatement().use { stmt ->
                stmt.execute("SET search_path TO \"$schema\"")
            }
        }
        log.debug { "Database connection schema set to '$schema'" }
    }
}
