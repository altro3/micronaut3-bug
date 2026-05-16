package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.MigrationVersion
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.core.io.Resource
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils
import org.springframework.jdbc.core.JdbcTemplate

open class FlywayRollbackEngine(
    private val flywayProperties: FlywayProperties,
    private val jdbcTemplate: JdbcTemplate,
    private val stepExecutor: FlywayRollbackStepExecutor,
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
     * Находит максимальный installed_rank миграций внутри этой папки и откатывает все последующие миграции.
     */
    open fun rollbackToTag(tag: String) {
        log.info { "Starting directory-based rollback to tag (target folder name): $tag" }

        // 1. Собираем абсолютно все SQL файлы из всех базовых локаций рекурсивно
        val allResources = cleanedLocations.flatMap { cleanLocation ->
            val pattern = "$CLASSPATH_PREFIX$cleanLocation/$SQL_ALL_PATTERN"
            try {
                resourceResolver.getResources(pattern).toList()
            } catch (e: Exception) {
                log.warn(e) { "Failed to scan location: $cleanLocation" }
                emptyList()
            }
        }

        // 2. Исправлено: Безопасное выделение имени конечной папки для IDE и для JAR сред
        val resourcesInTag = allResources.filter { resource ->
            try {
                // Получаем чистый абсолютный путь к файлу без префиксов 'file:' и скобок
                val pathString = resource.file.absolutePath.replace(BACKSLASH, SLASH)
                val tokens = pathString.split(SLASH).filter { it.isNotBlank() }

                if (tokens.size >= 2) {
                    val parentDir = tokens[tokens.size - 2]
                    parentDir == tag
                } else {
                    false
                }
            } catch (_: Exception) {
                // На случай, если это JAR среда и .file выбросит UnsupportedOperationException
                val urlPath = resource.url.toString().replace(BACKSLASH, SLASH)
                val cleanPath = urlPath.substringAfter("!").removeSuffix("]").removePrefix("[")
                val tokens = cleanPath.split(SLASH).filter { it.isNotBlank() }

                if (tokens.size >= 2) {
                    val parentDir = tokens[tokens.size - 2]
                    parentDir == tag
                } else {
                    false
                }
            }
        }

        if (resourcesInTag.isEmpty()) {
            throw IllegalArgumentException(
                "Tag directory '$tag' was not found or contains no migration files. " +
                        "Checked active root locations: $cleanedLocations"
            )
        }

        // 3. Вытаскиваем версии (таймстампы) файлов, принадлежащих этому тегу
        val versionsInTag = resourcesInTag.mapNotNull { resource ->
            val filename = resource.filename ?: return@mapNotNull null

            // Удаляем префиксы 'V' или 'U' в любом регистре, чтобы строка начиналась сразу с цифр
            val cleanFilename = filename.removePrefix("V").removePrefix("v")
                .removePrefix("U").removePrefix("u")

            VERSION_REGEX.find(cleanFilename)?.value
        }.toSet()

        if (versionsInTag.isEmpty()) {
            throw IllegalStateException("No valid versioned files found for tag directory '$tag'")
        }

        log.info { "Tag '$tag' resolved to versions: $versionsInTag" }

        // 4. Ищем максимальный installed_rank среди успешно примененных миграций этого тега
        val placeholders = versionsInTag.joinToString { "?" }
        val sql = "SELECT installed_rank FROM $historyTable WHERE version IN ($placeholders) AND success = true"
        val installedRanks = jdbcTemplate.queryForList(sql, Int::class.java, *versionsInTag.toTypedArray())

        if (installedRanks.isEmpty()) {
            log.info { "No migrations from tag '$tag' have been applied to the database yet. Nothing to roll back." }
            return
        }

        val maxRankInTag = installedRanks.maxOrNull()
            ?: throw IllegalStateException("Could not determine max installed_rank for tag '$tag'")

        log.info { "Tag '$tag' matched highest database installed_rank: $maxRankInTag" }

        // 5. Выбираем из истории ВСЕ миграции, которые были накатаны ПОЗДНЕЕ этого максимального ранга
        val migrationsToUndo = jdbcTemplate.queryForList(
            "SELECT version FROM $historyTable WHERE installed_rank > ? AND success = true ORDER BY installed_rank DESC",
            String::class.java,
            maxRankInTag
        ).filterNotNull()

        if (migrationsToUndo.isEmpty()) {
            log.info { "Database state is already at tag '$tag' (or below). No subsequent migrations found." }
            return
        }

        // 6. Запускаем последовательный откат через изолированный executor
        log.info { "Found ${migrationsToUndo.size} subsequent migration(s) to roll back." }
        migrationsToUndo.forEach { versionStr ->
            processVersionRollback(versionStr)
        }

        log.info { "Successfully rolled back all migrations applied after tag '$tag'" }
    }

    /**
     * Внутренний метод поиска ресурсов для конкретной версии миграции и отправки их на выполнение.
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

    /**
     * Получение списка примененных версий миграций в порядке убывания их выполнения.
     */
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
