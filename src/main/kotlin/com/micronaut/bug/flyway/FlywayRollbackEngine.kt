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
    flywayProperties: FlywayProperties,
    resourceLoader: ResourceLoader,
) {
    private val log = KotlinLogging.logger {}
    private val resourceResolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)

    private val rawHistoryTable: String
    private val activeSchema: String
    private val fullHistoryTablePath: String
    private val cleanedLocations: List<String>

    init {
        this.activeSchema = flywayProperties.schemas.firstOrNull() ?: ""
        this.rawHistoryTable = flywayProperties.table ?: DEFAULT_TABLE

        this.fullHistoryTablePath = if (activeSchema.isNotBlank()) {
            "$activeSchema.$rawHistoryTable"
        } else {
            rawHistoryTable
        }

        this.cleanedLocations = flywayProperties.locations.map {
            it.removePrefix(CLASSPATH_PREFIX).removePrefix(SLASH)
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

        // --- ШАГ ВАЛИДАЦИИ (Новый код) ---
        // Проверяем ВСЮ цепочку миграций, выбранных для отката, ДО начала выполнения SQL
        for (i in 0 until stepsToExecute) {
            val version = applied[i]
            val meta = resourcesMeta[version]
            if (meta?.isNoRollback == true || (meta?.undoResource == null && meta?.isNoRollback != true)) {
                throw IllegalStateException("Rollback aborted! Migration $version is marked as non-rollable (_norb) or is missing an undo script. Continuous rollback chain is broken.")
            }
        }

        for (currentStep in 0 until stepsToExecute) {
            processVersionRollback(applied[currentStep], resourcesMeta)
        }
        log.info { "Successfully executed $stepsToExecute programmatic rollback step(s)" }
    }

    open fun rollbackToVersion(targetVersion: String) {
        val target = MigrationVersion.fromVersion(targetVersion)
        log.info { "Starting programmatic rollback to target version: $targetVersion" }

        val applied = getAppliedMigrationsDesc()
        val toUndo = applied.filter { MigrationVersion.fromVersion(it) > target }

        if (toUndo.isEmpty()) {
            log.info { "Database state is already at or below version $targetVersion. No action required." }
            return
        }

        // Сканируем ресурсы один раз
        val resourcesMeta = scanAllMigrationResources()

        toUndo.forEach { versionStr ->
            processVersionRollback(versionStr, resourcesMeta)
        }
        log.info { "Successfully rolled back to version $targetVersion" }
    }

    open fun rollbackToTag(tag: String) {
        log.info { "Starting parallel-safe directory-based rollback to tag (target folder name): $tag" }

        val targetTagVersion = MigrationVersion.fromVersion(tag)
        val resourcesMeta = scanAllMigrationResources()

        // Проверяем, что целевой тег вообще физически существует в ресурсах проекта
        val availableFolders = resourcesMeta.values.map { it.parentFolder }.toSet()
        if (!availableFolders.contains(tag)) {
            throw IllegalArgumentException("Tag directory '$tag' was not found or contains no migration files.")
        }

        // Шаг 2. Извлекаем ВСЕ успешные примененные миграции из БД
        val appliedMigrations = jdbcTemplate.query(
            "SELECT version, installed_rank FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC"
        ) { rs, _ ->
            Pair(rs.getString("version"), rs.getInt("installed_rank"))
        }.filterNotNull()

        if (appliedMigrations.isEmpty()) {
            log.warn { "No applied migrations found in history table. Nothing to roll back." }
            return
        }

        // Шаг 3. Находим стартовую точку "чужого" (более нового) релиза.
        val minRankOfSubsequentReleases = appliedMigrations.mapNotNull { (version, rank) ->
            val folderName = resourcesMeta[version]?.parentFolder ?: return@mapNotNull null
            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                if (folderVersion > targetTagVersion) rank else null
            } catch (_: Exception) {
                null // Игнорируем не-версионные папки (например, common)
            }
        }.minOrNull()

        // Шаг 4. Отбираем только те миграции, которые подлежат уничтожению
        val migrationsToUndo = appliedMigrations.filter { (version, rank) ->
            val folderName = resourcesMeta[version]?.parentFolder

            if (folderName == null) {
                return@filter true // Если файла нет в коде, откатываем обязательно
            }

            if (minRankOfSubsequentReleases != null && rank >= minRankOfSubsequentReleases) {
                return@filter true // ПРАВИЛО 1: Накатили одновременно или позже нового релиза
            }

            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                folderVersion > targetTagVersion // ПРАВИЛО 2: Версия папки выше целевой
            } catch (e: Exception) {
                false // Папки типа common/callbacks сохраняем, если они ниже критического ранга
            }
        }.map { it.first }

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already aligned with tag '$tag'. No action required." }
            return
        }

        // Шаг 5. Запускаем последовательный откат
        log.info { "Parallel-safe mode: Found ${migrationsToUndo.size} subsequent migration(s) to roll back." }
        migrationsToUndo.forEach { versionStr ->
            processVersionRollback(versionStr, resourcesMeta)
        }

        log.info { "Successfully rolled back all subsequent migrations to align with tag '$tag'" }
    }

    private fun processVersionRollback(version: String, resourcesMeta: Map<String, MigrationResourceMeta>) {
        val meta = resourcesMeta[version]
        val scriptResource = meta?.undoResource
        val isNoRollbackMigration = meta?.isNoRollback ?: false

        if (scriptResource == null && !isNoRollbackMigration) {
            throw IllegalStateException(
                "Rollback failed! Neither U${version}__*.sql nor V${version}__*_norb.sql found in project resources."
            )
        }

        stepExecutor.executeRollbackStep(
            version = version,
            scriptResource = scriptResource,
            isNoRollbackMigration = isNoRollbackMigration,
            activeSchema = activeSchema,
            historyTable = rawHistoryTable,
        )
    }

    private fun getAppliedMigrationsDesc(): List<String> {
        return jdbcTemplate.queryForList(
            "SELECT version FROM $fullHistoryTablePath WHERE success = true ORDER BY installed_rank DESC",
            String::class.java
        ).filterNotNull()
    }

    /**
     * Сканирует ресурсы один раз, безопасно извлекая имена папок (включая работу внутри JAR)
     * и сопоставляя их с версиями.
     */
    private fun scanAllMigrationResources(): Map<String, MigrationResourceMeta> {
        val metaMap = mutableMapOf<String, MigrationResourceMeta>()
        val locations = cleanedLocations.ifEmpty { listOf("classpath:db/migration") }

        locations.forEach { cleanLocation ->
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

                // Удаляем строго заглавные префиксы V или U
                val cleanFilename = filename.replace(Regex("^[VU]"), "")
                val version = VERSION_REGEX.find(cleanFilename)?.value ?: return@forEach

                val currentMeta = metaMap.computeIfAbsent(version) { MigrationResourceMeta(parentFolder = parentFolder) }

                // Обработка строго по заглавным буквам
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

    /**
     * Внутренний контейнер метаданных для файла миграции
     */
    private class MigrationResourceMeta(
        val parentFolder: String,
        var undoResource: Resource? = null,
        var isNoRollback: Boolean = false
    )

    companion object {
        private const val DEFAULT_TABLE = "flyway_schema_history"
        private const val CLASSPATH_PREFIX = "classpath:"
        private const val SLASH = "/"
        private const val BACKSLASH = "\\"
        private const val SQL_ALL_PATTERN = "**/*.sql"
        private const val NO_ROLLBACK_SUFFIX = "_norb"
        private val VERSION_REGEX = Regex("""^\d+""")
    }
}
