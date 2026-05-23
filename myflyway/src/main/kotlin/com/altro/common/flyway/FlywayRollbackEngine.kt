package com.altro.common.flyway

import com.altro.common.flyway.FlywayConventionConst.DEFAULT_TABLE
import com.altro.common.flyway.FlywayConventionConst.NO_ROLLBACK_SUFFIX
import com.altro.common.flyway.FlywayMetadataResolver.MigrationResourceMeta
import com.altro.common.flyway.config.FlywayRollbackProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.MigrationVersion
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.jdbc.core.JdbcTemplate
import javax.sql.DataSource

open class FlywayRollbackEngine(
    private val dataSource: DataSource,
    private val flywayRollbackProperties: FlywayRollbackProperties,
    private val metadataResolver: FlywayMetadataResolver,
    flywayProperties: FlywayProperties,
) {
    private val log = KotlinLogging.logger {}

    private val jdbcTemplate = JdbcTemplate(dataSource)

    private val rawHistoryTable: String = validateIdentifier(flywayProperties.table ?: DEFAULT_TABLE)
    private val activeSchema: String = validateIdentifier(flywayProperties.schemas.firstOrNull() ?: "")
    private val fullHistoryTablePath: String = if (activeSchema.isNotBlank()) """"${activeSchema.trim()}"."${rawHistoryTable.trim()}"""" else """"${rawHistoryTable.trim()}""""

    open fun rollback(steps: Int = 1) {
        require(steps > 0) { "Rollback steps count must be greater than 0. Passed: $steps" }
        log.info { "Starting programmatic rollback for $steps step(s)" }

        val applied = getAppliedMigrationsDesc()
        if (applied.isEmpty()) {
            log.warn { "No applied migrations found in $fullHistoryTablePath. Nothing to roll back." }
            return
        }

        val resourcesMeta = metadataResolver.getResolvedMetaMap()
        val stepsToExecute = minOf(steps, applied.size)

        // Вариант А: Защитная сквозная Fail-Fast проверка всей цепочки до начала выполнения роллбэков
        validateRollbackChainOrThrow(applied, resourcesMeta)

        val versionsToRollback = applied.take(stepsToExecute)
        executeInRollbackSession(versionsToRollback, resourcesMeta)
        log.info { "Successfully executed $stepsToExecute programmatic rollback step(s)" }
    }

    open fun rollbackToVersion(targetVersion: String) {
        val target = MigrationVersion.fromVersion(targetVersion)
        log.info { "Starting programmatic rollback to target version: $targetVersion" }

        val applied = getAppliedMigrationsDesc()
        val toUndo = applied.filter { versionStr ->
            MigrationVersion.fromVersion(versionStr) > target
        }

        if (toUndo.isEmpty()) {
            log.info { "Database state is already at or below version $targetVersion. No action required." }
            return
        }

        val resourcesMeta = metadataResolver.getResolvedMetaMap()
        validateRollbackChainOrThrow(applied, resourcesMeta)

        executeInRollbackSession(toUndo, resourcesMeta)
        log.info { "Successfully rolled back to version $targetVersion" }
    }

    open fun rollbackToTag(tag: String) {
        log.info { "Starting directory-based rollback to target release folder name: $tag" }

        val targetTagVersion = MigrationVersion.fromVersion(tag)
        val resourcesMeta = metadataResolver.getResolvedMetaMap()

        val availableFolders = resourcesMeta.values.map { it.releaseTag }.toSet()
        if (!availableFolders.contains(tag)) {
            throw IllegalArgumentException("Tag directory '$tag' was not found or contains no migration files.")
        }

        val appliedMigrations = jdbcTemplate.query(
            "SELECT version, installed_rank FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC"
        ) { rs, _ ->
            Pair(rs.getString("version"), rs.getInt("installed_rank"))
        }.filterNotNull()

        if (appliedMigrations.isEmpty()) {
            log.warn { "No applied migrations found in history table. Nothing to roll back." }
            return
        }

        val minRankOfSubsequentReleases = appliedMigrations.mapNotNull { (version, rank) ->
            val tagName = resourcesMeta[version]?.releaseTag ?: return@mapNotNull null
            try {
                val folderVersion = MigrationVersion.fromVersion(tagName)
                if (folderVersion > targetTagVersion) rank else null
            } catch (_: Exception) {
                null
            }
        }.minOrNull()

        val migrationsToUndo = appliedMigrations.filter { (version, rank) ->
            val tagName = resourcesMeta[version]?.releaseTag ?: return@filter false

            if (minRankOfSubsequentReleases != null && rank >= minRankOfSubsequentReleases) {
                return@filter true
            }

            val folderVersion = try {
                MigrationVersion.fromVersion(tagName)
            } catch (_: Exception) {
                return@filter false
            }

            folderVersion > targetTagVersion
        }.map { it.first }

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already aligned with tag '$tag'. No action required." }
            return
        }

        validateRollbackChainOrThrow(appliedMigrations.map { it.first }, resourcesMeta)

        executeInRollbackSession(migrationsToUndo, resourcesMeta)
        log.info { "Successfully rolled back all subsequent migrations to align with tag '$tag'" }
    }

    private fun validateRollbackChainOrThrow(versions: List<String>, resourcesMeta: Map<String, MigrationResourceMeta>) {
        for (version in versions) {
            val meta = resourcesMeta[version]
            if (meta == null) {
                throw IllegalStateException("Rollback validation failed! Version $version is applied in the database, but no local migration files were found.")
            }

            if (meta.isNoRollback) {
                if (!flywayRollbackProperties.force) {
                    throw IllegalStateException("Rollback aborted! Migration $version is explicitly marked as non-rollable ($NO_ROLLBACK_SUFFIX). To force bypass, enable 'spring.flyway.rollback.force=true'.")
                } else {
                    log.warn { "Migration $version is marked as $NO_ROLLBACK_SUFFIX, but 'force' flag is ENABLED. Bypassing." }
                }
            }
        }
    }

    private fun executeInRollbackSession(versions: List<String>, resourcesMeta: Map<String, MigrationResourceMeta>) {
        if (versions.isEmpty()) return

        dataSource.connection.use { conn ->
            val stepExecutor = FlywayRollbackStepExecutor(conn)

            versions.forEach { version ->
                val meta = resourcesMeta[version] ?: throw IllegalStateException("Metadata missing for version $version.")
                stepExecutor.executeRollbackStep(
                    version = version,
                    scriptResource = meta.undoResource,
                    isNoRollbackMigration = meta.isNoRollback,
                    activeSchema = activeSchema,
                    historyTable = rawHistoryTable
                )
            }
        }
        log.info { "Successfully executed sequential rollback for ${versions.size} migration(s) inside a single connection session." }
    }

    private fun getAppliedMigrationsDesc(): List<String> =
        jdbcTemplate.queryForList(
            "SELECT version FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC",
            String::class.java
        ).filterNotNull()

    private fun validateIdentifier(identifier: String): String {
        val trimmed = identifier.trim()
        val validPattern = Regex("^[a-zA-Z0-9_]+$")
        if (trimmed.isNotBlank() && !validPattern.matches(trimmed)) {
            throw IllegalArgumentException("CRITICAL: Invalid database identifier detected: '$trimmed'.")
        }
        return trimmed
    }
}
