package com.altro.common.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.configuration.ClassicConfiguration
import org.flywaydb.core.internal.jdbc.JdbcConnectionFactory
import org.flywaydb.core.internal.parser.ParsingContext
import org.flywaydb.core.internal.resource.StringResource
import org.flywaydb.database.postgresql.PostgreSQLDatabase
import org.flywaydb.database.postgresql.PostgreSQLParser
import org.springframework.core.io.Resource
import org.springframework.jdbc.datasource.SingleConnectionDataSource
import java.nio.charset.StandardCharsets
import java.sql.Connection

open class FlywayRollbackStepExecutor(
    private val conn: Connection,
) {

    private val log = KotlinLogging.logger {}
    private val postgresParser: PostgreSQLParser

    init {
        val classicConfig = ClassicConfiguration()
        val singleConnectionDataSource = SingleConnectionDataSource(conn, true)
        val connectionFactory = JdbcConnectionFactory(
            singleConnectionDataSource,
            classicConfig,
            null,
        )
        val database = PostgreSQLDatabase(classicConfig, connectionFactory, null)
        val parsingContext = ParsingContext().apply { this.database = database }

        postgresParser = PostgreSQLParser(classicConfig, parsingContext)
    }

    open fun executeRollbackStep(
        version: String,
        scriptResource: Resource?,
        isNoRollbackMigration: Boolean,
        activeSchema: String,
        historyTable: String
    ) {
        log.info { "Processing atomic rollback step for version: $version" }
        val fullTableName = if (activeSchema.isNotBlank()) "\"$activeSchema\".\"$historyTable\"" else "\"$historyTable\""

        // Настраиваем схему в рамках текущей сессии
        if (activeSchema.isNotBlank()) {
            try {
                conn.schema = activeSchema
            } catch (_: Exception) {
                conn.createStatement().use { it.execute("SET search_path TO \"$activeSchema\"") }
            }
        }

        // Включаем режим ручного управления транзакцией для всего файла
        conn.autoCommit = false

        try {
            if (isNoRollbackMigration) {
                log.info { "Migration $version is explicitly marked as non-rollable. Skipping SQL execution." }
            } else {
                val resource = scriptResource
                    ?: throw IllegalStateException("Rollback failed! No undo script resource provided for version $version")
                log.info { "Found native-style undo script: ${resource.filename}. Parsing and executing..." }

                val sqlContent = resource.getContentAsString(StandardCharsets.UTF_8)
                val statementReader = postgresParser.parse(StringResource(sqlContent))

                while (statementReader.hasNext()) {
                    val statement = statementReader.next() ?: continue
                    val sqlToExecute = statement.sql ?: continue
                    if (sqlToExecute.isBlank()) continue

                    // Если стейтмент требует автокоммита (например, CONCURRENTLY),
                    // а у нас строгое правило "один файл - одна транзакция", прерываем выполнение.
                    if (!statement.canExecuteInTransaction()) {
                        throw IllegalStateException(
                            "Statement in version $version cannot execute inside a transaction (e.g. CREATE INDEX CONCURRENTLY). " +
                                    "This violates the 'one file - one transaction' constraint. Statement: $sqlToExecute"
                        )
                    }

                    log.info {
                        """
                        |
                        |Executing SQL statement:
                        |--------------------------------------------------
                        |${sqlToExecute.trim()}
                        |--------------------------------------------------
                        """.trimMargin()
                    }
                    conn.createStatement().use { it.execute(sqlToExecute) }
                }
            }

            // Фиксируем транзакцию выполнения скрипта ТУТ (один раз на весь файл)
            conn.commit()
            log.info { "SQL script for version $version successfully committed." }

        } catch (e: Exception) {
            // Если упал любой стейтмент скрипта — откатываем ВСЕ изменения этого файла
            conn.rollback()
            log.error(e) { "Rollback SQL execution failed for version $version. Whole transaction rolled back." }
            throw e
        }

        // Вторая транзакция — обновление метаданных (выполняется только при успехе первой)
        try {
            conn.prepareStatement("DELETE FROM $fullTableName WHERE version = ?").use { stmt ->
                stmt.setString(1, version)
                if (stmt.executeUpdate() == 0) {
                    throw IllegalStateException("SQL executed, but failed to remove version $version from metadata table $fullTableName")
                }
            }
            conn.commit()
            log.info { "Metadata successfully synchronized for version $version." }
        } catch (e: Exception) {
            conn.rollback()
            log.error(e) { "Failed to update metadata table for version $version." }
            throw e
        }
    }
}
