package com.micronaut.bug.flyway

import com.micronaut.bug.flyway.config.FlywayRollbackProperties
import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.MigrationVersion
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.core.io.Resource
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils
import org.springframework.jdbc.core.JdbcTemplate

open class FlywayRollbackEngine(
    private val jdbcTemplate: JdbcTemplate,
    private val stepExecutor: FlywayRollbackStepExecutor,
    private val flywayProperties: FlywayProperties,
    private val flywayRollbackProperties: FlywayRollbackProperties,
    resourceLoader: ResourceLoader,
) {
    private val log = KotlinLogging.logger {}
    private val resourceResolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)

    private val rawHistoryTable: String = flywayProperties.table ?: DEFAULT_TABLE
    private val activeSchema: String = flywayProperties.schemas.firstOrNull() ?: ""
    private val fullHistoryTablePath: String = if (activeSchema.isNotBlank()) """"${activeSchema.trim()}"."${rawHistoryTable.trim()}"""" else """"${rawHistoryTable.trim()}""""

    open fun rollback(steps: Int = 1) {
        require(steps > 0) { "Rollback steps count must be greater than 0. Passed: $steps" }
        log.info { "Starting programmatic rollback for $steps step(s)" }

        val applied = getAppliedMigrationsDesc()
        if (applied.isEmpty()) {
            log.warn { "No applied migrations found in $fullHistoryTablePath. Nothing to roll back." }
            return
        }

        val resourcesMeta = scanAllMigrationResources()
        val stepsToExecute = minOf(steps, applied.size)

        val versionsToRollback = applied.take(stepsToExecute)
        validateRollbackChainOrThrow(versionsToRollback, resourcesMeta)

        for (currentStep in 0 until stepsToExecute) {
            processVersionRollback(applied[currentStep], resourcesMeta)
        }
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

        val resourcesMeta = scanAllMigrationResources()

        validateRollbackChainOrThrow(toUndo, resourcesMeta)

        toUndo.forEach { versionStr ->
            processVersionRollback(versionStr, resourcesMeta)
        }
        log.info { "Successfully rolled back to version $targetVersion" }
    }

    open fun rollbackToTag(tag: String) {
        log.info { "Starting parallel-safe directory-based rollback to tag (target folder name): $tag" }

        val targetTagVersion = MigrationVersion.fromVersion(tag)
        val resourcesMeta = scanAllMigrationResources()

        val availableFolders = resourcesMeta.values.map { it.parentFolder }.toSet()
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
            val folderName = resourcesMeta[version]?.parentFolder ?: return@mapNotNull null
            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                if (folderVersion > targetTagVersion) rank else null
            } catch (_: Exception) {
                null
            }
        }.minOrNull()

        val migrationsToUndo = appliedMigrations.filter { (version, rank) ->
            val folderName = resourcesMeta[version]?.parentFolder
                ?: return@filter false // Если файла нет в локальном classpath, мы не имеем права его откатывать!

            // Если ранг миграции выше критического ранга перемешанных релизов — берем под нож
            if (minRankOfSubsequentReleases != null && rank >= minRankOfSubsequentReleases) {
                return@filter true
            }

            // Проверяем, является ли папка валидным тегом-версией
            val folderVersion = try {
                MigrationVersion.fromVersion(folderName)
            } catch (_: Exception) {
                // Если папка называется "db", "migration" или "init" — это не наш тег релиза, игнорируем
                return@filter false
            }

            // Если версия папки строго выше целевого тега — её нужно откатить
            folderVersion > targetTagVersion
        }.map { it.first }

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already aligned with tag '$tag'. No action required." }
            return
        }

        validateRollbackChainOrThrow(migrationsToUndo, resourcesMeta)

        log.info { "Parallel-safe mode: Found ${migrationsToUndo.size} subsequent migration(s) to roll back." }
        migrationsToUndo.forEach { versionStr ->
            processVersionRollback(versionStr, resourcesMeta)
        }

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
                    throw IllegalStateException(
                        "Rollback aborted! Migration $version is explicitly marked as non-rollable (_norb). " +
                                "Continuous rollback chain is broken. To force bypass this migration and continue rolling back " +
                                "subsequent schemas, enable 'spring.flyway.rollback.force=true'."
                    )
                } else {
                    log.warn { "Migration $version is marked as _norb, but 'force' flag is ENABLED. Chain validation bypassed for this step." }
                }
            }

            if (!meta.isNoRollback && meta.undoResource == null) {
                throw IllegalStateException(
                    "Rollback aborted! Migration $version is missing an undo script (U${version}__*.sql) " +
                            "and is not marked as '$NO_ROLLBACK_SUFFIX'. Continuous rollback chain is broken."
                )
            }
        }
    }

    private fun processVersionRollback(version: String, resourcesMeta: Map<String, MigrationResourceMeta>) {

        val meta = resourcesMeta[version]
            ?: throw IllegalStateException("Rollback failed! Metadata missing for version $version.")

        if (meta.undoResource == null && (!meta.isNoRollback || !flywayRollbackProperties.force)) {
            throw IllegalStateException(
                "Rollback failed! Script resource is missing for version $version. " +
                        "If this is a non-rollable migration, ensure it has the '$NO_ROLLBACK_SUFFIX' suffix " +
                        "and restart the application with 'spring.flyway.rollback.force=true'."
            )
        }
        stepExecutor.executeRollbackStep(
            version = version,
            scriptResource = meta.undoResource,
            isNoRollbackMigration = meta.isNoRollback,
            activeSchema = activeSchema,
            historyTable = rawHistoryTable
        )
    }

    private fun getAppliedMigrationsDesc(): List<String> =
        jdbcTemplate.queryForList(
            "SELECT version FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC",
            String::class.java
        ).filterNotNull()

    private fun scanAllMigrationResources(): Map<String, MigrationResourceMeta> {
        val metaMap = mutableMapOf<String, MigrationResourceMeta>()
        val locations = flywayProperties.locations.ifEmpty { listOf(DEFAULT_LOCATION) }

        locations.forEach { rawLocation ->
            val cleanLocation = rawLocation.removeSuffix(SLASH)
            val pattern = if (cleanLocation.contains(":")) {
                "$cleanLocation/$SQL_ALL_PATTERN"
            } else {
                "$CLASSPATH_PREFIX$cleanLocation/$SQL_ALL_PATTERN"
            }

            val resources = try {
                resourceResolver.getResources(pattern)
            } catch (e: Exception) {
                log.warn(e) { "Failed to scan location: $cleanLocation" }
                emptyArray()
            }

            resources.forEach { resource ->
                val filename = resource.filename ?: return@forEach

                val urlPath = resource.url.toString().replace(BACKSLASH, SLASH)
                val tokens = urlPath.split(SLASH).filter { it.isNotBlank() }
                if (tokens.size < 2) return@forEach
                val parentFolder = tokens[tokens.size - 2]

                val cleanFilename = filename.replace(Regex("^[VU]"), "")
                val version = VERSION_REGEX.find(cleanFilename)?.value ?: return@forEach

                val currentMeta = metaMap.computeIfAbsent(version) { MigrationResourceMeta(parentFolder = parentFolder) }

                if (filename.startsWith("U")) {
                    currentMeta.undoResource = resource
                } else if (filename.startsWith("V")) {
                    // Явно перезаписываем parentFolder, так как тег релиза определяется именно по V-миграции!
                    currentMeta.parentFolder = parentFolder
                    if (filename.contains(NO_ROLLBACK_SUFFIX)) {
                        currentMeta.isNoRollback = true
                    }
                }
            }
        }
        return metaMap
    }

    private class MigrationResourceMeta(
        var parentFolder: String,
        var undoResource: Resource? = null,
        var isNoRollback: Boolean = false
    )

    companion object {
        private const val DEFAULT_LOCATION = "classpath:db/migration"
        private const val DEFAULT_TABLE = "flyway_schema_history"
        private const val CLASSPATH_PREFIX = "classpath:"
        private const val SLASH = "/"
        private const val BACKSLASH = "\\"
        private const val SQL_ALL_PATTERN = "**/*.sql"
        private const val NO_ROLLBACK_SUFFIX = "_norb"
        private val VERSION_REGEX = Regex("""^\d+""")
    }
}
