package com.micronaut.bug.flyway

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
    resourceLoader: ResourceLoader,
) {
    private val log = KotlinLogging.logger {}
    private val resourceResolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)

    private val rawHistoryTable: String
    private val activeSchema: String
    private val fullHistoryTablePath: String

    init {
        this.activeSchema = flywayProperties.schemas.firstOrNull() ?: ""
        this.rawHistoryTable = flywayProperties.table ?: DEFAULT_TABLE

        this.fullHistoryTablePath = if (activeSchema.isNotBlank()) {
            "$activeSchema.$rawHistoryTable"
        } else {
            rawHistoryTable
        }
    }

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

        // --- ВАЛИДАЦИЯ ПЕРЕД ЗАПУСКОМ ---
        val versionsToRollback = applied.take(stepsToExecute)
        validateRollbackChainOrThrow(versionsToRollback, resourcesMeta)

        // Если цепочка валидна — выполняем последовательный откат
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
            MigrationVersion.fromVersion(versionStr).compareTo(target) > 0
        }

        if (toUndo.isEmpty()) {
            log.info { "Database state is already at or below version $targetVersion. No action required." }
            return
        }

        val resourcesMeta = scanAllMigrationResources()

        // --- ВАЛИДАЦИЯ ПЕРЕД ЗАПУСКОМ ---
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
                if (folderVersion.compareTo(targetTagVersion) > 0) rank else null
            } catch (e: Exception) {
                null
            }
        }.minOrNull()

        val migrationsToUndo = appliedMigrations.filter { (version, rank) ->
            val folderName = resourcesMeta[version]?.parentFolder

            if (folderName == null) {
                return@filter true
            }

            if (minRankOfSubsequentReleases != null && rank >= minRankOfSubsequentReleases) {
                return@filter true
            }

            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                folderVersion.compareTo(targetTagVersion) > 0
            } catch (e: Exception) {
                false
            }
        }.map { it.first }

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already aligned with tag '$tag'. No action required." }
            return
        }

        // --- ВАЛИДАЦИЯ ПЕРЕД ЗАПУСКОМ ---
        validateRollbackChainOrThrow(migrationsToUndo, resourcesMeta)

        log.info { "Parallel-safe mode: Found ${migrationsToUndo.size} subsequent migration(s) to roll back." }
        migrationsToUndo.forEach { versionStr ->
            processVersionRollback(versionStr, resourcesMeta)
        }

        log.info { "Successfully rolled back all subsequent migrations to align with tag '$tag'" }
    }

    /**
     * Проверяет список версий на возможность отката до внесения изменений в БД.
     * Если найдена миграция без U-скрипта или с суффиксом _norb — выбрасывает исключение.
     */
    private fun validateRollbackChainOrThrow(versions: List<String>, resourcesMeta: Map<String, MigrationResourceMeta>) {
        for (version in versions) {
            val meta = resourcesMeta[version]

            if (meta?.isNoRollback == true) {
                throw IllegalStateException(
                    "Rollback aborted! Migration $version is explicitly marked as non-rollable (_norb). " +
                            "Continuous rollback chain is broken. Manual database intervention required."
                )
            }

            if (meta?.undoResource == null) {
                throw IllegalStateException(
                    "Rollback aborted! Migration $version is missing an undo script (U${version}__*.sql) " +
                            "and is not marked as '_norb'. Continuous rollback chain is broken."
                )
            }
        }
    }

    private fun processVersionRollback(version: String, resourcesMeta: Map<String, MigrationResourceMeta>) {
        val meta = resourcesMeta[version]
        val scriptResource = meta?.undoResource
            ?: throw IllegalStateException("Rollback failed! Script resource disappeared for version $version.")

        stepExecutor.executeRollbackStep(
            version = version,
            scriptResource = scriptResource,
            isNoRollbackMigration = false, // После валидации здесь всегда гарантирован настоящий откат
            activeSchema = activeSchema,
            historyTable = rawHistoryTable
        )
    }

    private fun getAppliedMigrationsDesc(): List<String> {
        return jdbcTemplate.queryForList(
            "SELECT version FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC",
            String::class.java
        ).filterNotNull()
    }

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
                    if (filename.contains(NO_ROLLBACK_SUFFIX)) {
                        currentMeta.isNoRollback = true
                    }
                }
            }
        }
        return metaMap
    }

    private class MigrationResourceMeta(
        val parentFolder: String,
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
