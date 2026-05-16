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

    private val historyTable: String
    private val activeSchema: String
    private val cleanedLocations: List<String>

    init {
        val schemaName = flywayProperties.schemas.firstOrNull() ?: DEFAULT_SCHEMA
        val tableName = flywayProperties.table ?: DEFAULT_TABLE

        this.activeSchema = schemaName
        this.historyTable = "$activeSchema.$tableName"
        this.cleanedLocations = flywayProperties.locations.map {
            it.removePrefix(CLASSPATH_PREFIX).removePrefix(SLASH)
        }
    }

    /**
     * Откат на указанное количество шагов назад.
     */
    open fun rollback(steps: Int = 1) {
        require(steps > 0) { "Rollback steps count must be greater than 0. Passed: $steps" }
        log.info { "Starting programmatic rollback for $steps step(s)" }

        val applied = getAppliedMigrationsDesc()
        if (applied.isEmpty()) {
            log.warn { "No applied migrations found in $historyTable. Nothing to roll back." }
            return
        }

        val stepsToExecute = minOf(steps, applied.size)
        for (currentStep in 0 until stepsToExecute) {
            processVersionRollback(applied[currentStep])
        }
        log.info { "Successfully executed $stepsToExecute programmatic rollback step(s)" }
    }

    /**
     * Откат до определенной версии (все версии строго выше целевой будут удалены).
     */
    open fun rollbackToVersion(targetVersion: String) {
        val target = MigrationVersion.fromVersion(targetVersion)
        log.info { "Starting programmatic rollback to target version: $targetVersion" }

        val applied = getAppliedMigrationsDesc()
        val toUndo = applied.filter { MigrationVersion.fromVersion(it) > target }

        if (toUndo.isEmpty()) {
            log.info { "Database state is already at or below version $targetVersion. No action required." }
            return
        }

        toUndo.forEach { versionStr ->
            processVersionRollback(versionStr)
        }
        log.info { "Successfully rolled back to version $targetVersion" }
    }

    /**
     * Откат по имени конечной папки (тега).
     * Безопасен при параллельной разработке нескольких релизных веток и наличии сервисных папок (common, callbacks).
     */
    open fun rollbackToTag(tag: String) {
        log.info { "Starting parallel-safe directory-based rollback to tag (target folder name): $tag" }

        val targetTagVersion = MigrationVersion.fromVersion(tag)
        val allResources = cleanedLocations.flatMap { cleanLocation ->
            val pattern = "$CLASSPATH_PREFIX$cleanLocation/$SQL_ALL_PATTERN"
            try {
                resourceResolver.getResources(pattern).toList()
            } catch (e: Exception) {
                log.warn(e) { "Failed to scan location: $cleanLocation" }
                emptyList()
            }
        }

        // Шаг 1. Строим карту: Версия миграции -> Имя её родительской папки
        val fileToFolderMap = mutableMapOf<String, String>()

        allResources.forEach { resource ->
            val filename = resource.filename ?: return@forEach

            // Безопасно вытаскиваем цифры версии, удаляя префиксы V/U в любом регистре
            val cleanFilename = filename.replace(Regex("^[VvUu]"), "")
            val version = VERSION_REGEX.find(cleanFilename)?.value ?: return@forEach

            // Безопасно определяем имя папки (работает и в IDE, и внутри собранного JAR)
            val cleanPath = try {
                resource.file.absolutePath.replace(BACKSLASH, SLASH)
            } catch (_: Exception) {
                val urlPath = resource.url.toString().replace(BACKSLASH, SLASH)
                if (urlPath.contains("!")) urlPath.substringAfter("!") else urlPath
            }

            val tokens = cleanPath.split(SLASH).filter { it.isNotBlank() }

            if (tokens.size >= 2) {
                val parentDir = tokens[tokens.size - 2]
                fileToFolderMap[version] = parentDir
            }
        }

        // Проверяем, что целевой тег вообще физически существует в ресурсах проекта
        if (!fileToFolderMap.values.contains(tag)) {
            throw IllegalArgumentException("Tag directory '$tag' was not found or contains no migration files.")
        }

        // Шаг 2. Извлекаем ВСЕ успешные примененные миграции из БД
        val appliedMigrations = jdbcTemplate.query(
            "SELECT version, installed_rank FROM $historyTable WHERE success = true ORDER BY installed_rank DESC"
        ) { rs, _ ->
            Pair(rs.getString("version"), rs.getInt("installed_rank"))
        }.filterNotNull()

        if (appliedMigrations.isEmpty()) {
            log.warn { "No applied migrations found in history table. Nothing to roll back." }
            return
        }

        // Шаг 3. Находим стартовую точку "чужого" (более нового) релиза.
        // Ищем минимальный installed_rank среди миграций, папки которых строго старше нашего тега.
        val minRankOfSubsequentReleases = appliedMigrations.mapNotNull { (version, rank) ->
            val folderName = fileToFolderMap[version] ?: return@mapNotNull null
            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                if (folderVersion > targetTagVersion) rank else null
            } catch (e: Exception) {
                // Игнорируем папки типа 'common', так как они не парсятся в объект версии
                null
            }
        }.minOrNull()

        // Шаг 4. Отбираем только те миграции, которые подлежат уничтожению
        val migrationsToUndo = appliedMigrations.filter { (version, rank) ->
            val folderName = fileToFolderMap[version]

            // Если миграции нет в текущих ресурсах (например, её удалили из кода) — откатываем обязательно
            if (folderName == null) {
                return@filter true
            }

            // ПРАВИЛО 1: Если в базе уже начался более новый релиз, то абсолютно ВСЕ миграции,
            // накатаные одновременно с ним или позже него (включая папки 'common'), идут под нож.
            if (minRankOfSubsequentReleases != null && rank >= minRankOfSubsequentReleases) {
                return@filter true
            }

            // ПРАВИЛО 2: Если до старта нового релиза дело не дошло, проверяем версию папки напрямую.
            // Миграции из будущих релизов (например, '2.0' при откате до '1.1') — откатываем.
            try {
                val folderVersion = MigrationVersion.fromVersion(folderName)
                folderVersion > targetTagVersion
            } catch (e: Exception) {
                // Если имя папки не версия (common, callbacks) и её rank меньше критического,
                // значит она создана в эпоху целевого тега или раньше — мы её сохраняем (false)
                false
            }
        }.map { it.first }

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already aligned with tag '$tag'. No action required." }
            return
        }

        // Шаг 5. Запускаем последовательный откат строго сверху вниз (от больших рангов к меньшим)
        log.info { "Parallel-safe mode: Found ${migrationsToUndo.size} subsequent migration(s) to roll back." }
        migrationsToUndo.forEach { versionStr ->
            processVersionRollback(versionStr)
        }

        log.info { "Successfully rolled back all subsequent migrations to align with tag '$tag'" }
    }

    /**
     * Поиск ресурсов миграции и отправка их на изолированное транзакционное исполнение.
     */
    private fun processVersionRollback(version: String) {
        var scriptResource: Resource? = null
        var isNoRollbackMigration = false

        for (cleanLocation in cleanedLocations) {
            val undoPattern = "$CLASSPATH_PREFIX$cleanLocation/**/U${version}__*.sql"
            val norbPattern = "$CLASSPATH_PREFIX$cleanLocation/**/V${version}__*_norb.sql"

            val undoResources = resourceResolver.getResources(undoPattern)
            if (undoResources.isNotEmpty()) {
                scriptResource = undoResources.first()
                break
            }

            val norbResources = resourceResolver.getResources(norbPattern)
            if (norbResources.isNotEmpty()) {
                isNoRollbackMigration = true
                break
            }
        }

        if (scriptResource == null && !isNoRollbackMigration) {
            throw IllegalStateException("Rollback failed! Neither U${version}__*.sql nor V${version}__*_norb.sql found.")
        }

        stepExecutor.executeRollbackStep(
            version = version,
            scriptResource = scriptResource,
            isNoRollbackMigration = isNoRollbackMigration,
            activeSchema = activeSchema,
            historyTable = historyTable
        )
    }

    private fun getAppliedMigrationsDesc(): List<String> {
        return jdbcTemplate.queryForList(
            "SELECT version FROM $historyTable WHERE success = true ORDER BY installed_rank DESC",
            String::class.java
        ).filterNotNull()
    }

    companion object {
        private const val DEFAULT_SCHEMA = "public"
        private const val DEFAULT_TABLE = "flyway_schema_history"
        private const val CLASSPATH_PREFIX = "classpath:"
        private const val SLASH = "/"
        private const val BACKSLASH = "\\"
        private const val SQL_ALL_PATTERN = "**/*.sql"
        private val VERSION_REGEX = Regex("""^\d+""")
    }
}
