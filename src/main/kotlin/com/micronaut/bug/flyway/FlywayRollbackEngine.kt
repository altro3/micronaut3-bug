package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.Flyway
import org.flywaydb.core.api.MigrationVersion
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.io.BufferedReader

open class FlywayRollbackEngine(
    private val flyway: Flyway,
    private val jdbcTemplate: JdbcTemplate,
    private val resourceLoader: ResourceLoader,
    private val locations: List<String>
) {

    private val log = KotlinLogging.logger {}
    private val historyTable: String
    private val activeSchema: String

    init {
        val schemaName = flyway.configuration.schemas.firstOrNull() ?: "public"
        val tableName = flyway.configuration.table
        this.activeSchema = schemaName
        this.historyTable = "$schemaName.$tableName"
    }

    /**
     * Эквивалент Liquibase: rollback <count>
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    open fun rollback(steps: Int = 1) {
        require(steps > 0) { "Rollback steps count must be greater than 0. Passed: $steps" }
        log.info { "Starting Community-style programmatic rollback for $steps step(s)" }

        val applied = getAppliedMigrationsDesc()
        if (applied.isEmpty()) {
            log.warn { "No applied migrations found in $historyTable. Nothing to roll back." }
            return
        }

        val stepsToExecute = minOf(steps, applied.size)
        try {
            repeat(stepsToExecute) { currentStep ->
                val targetVersion = applied[currentStep]
                executeRollbackScript(targetVersion)
            }
            log.info { "Successfully executed $stepsToExecute programmatic rollback step(s)" }
        } catch (e: Exception) {
            log.error(e) { "Rollback sequence failed." }
            throw e
        }
    }

    /**
     * Эквивалент Liquibase: rollback <version>
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    open fun rollbackToVersion(targetVersion: String) {
        val target = MigrationVersion.fromVersion(targetVersion)
        log.info { "Starting Community-style programmatic rollback to target version: $targetVersion" }

        val applied = getAppliedMigrationsDesc()
        val toUndo = applied.filter { MigrationVersion.fromVersion(it) > target }

        if (toUndo.isEmpty()) {
            log.info { "Database state is already at or below version $targetVersion. No action required." }
            return
        }

        try {
            toUndo.forEach { versionStr ->
                executeRollbackScript(versionStr)
            }
            log.info { "Successfully rolled back to version $targetVersion" }
        } catch (e: Exception) {
            log.error(e) { "Rollback to version $targetVersion failed" }
            throw e
        }
    }

    /**
     * Эквивалент Liquibase: rollback <tag>
     * В качестве тега используется имя директории релиза в любом формате (например, "2.0", "v2-hotfix").
     * Откатывает все миграции, которые были применены ПОСЛЕ этого релиза.
     *
     * @param tag Название папки релиза (любой формат).
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    open fun rollbackToTag(tag: String) {
        log.info { "Starting directory-based rollback to tag (release directory): $tag" }

        // Находим точный путь к папке релиза среди активных локаций Flyway
        // Сработает для "classpath:db/migration/2.0" и для "classpath:db/migration/v2-hotfix" одинаково успешно
        val targetLocation = locations.firstOrNull { it.endsWith("/$tag") }
            ?: throw IllegalArgumentException(
                "Tag directory '$tag' was not found among active Flyway locations: $locations. " +
                        "Make sure this directory exists and contains at least one migration file."
            )

        val resolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)
        val pattern = "$targetLocation/**/*.sql"

        val resources = try {
            resolver.getResources(pattern)
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to scan directory for tag '$tag' using pattern '$pattern'", e)
        }

        if (resources.isEmpty()) {
            throw IllegalArgumentException("No migration files found in release directory (tag) '$tag' by pattern '$pattern'")
        }

        val versionsInTag = resources.mapNotNull { resource ->
            val filename = resource.filename ?: return@mapNotNull null
            Regex("""^\d+""").find(filename)?.value
        }.map { MigrationVersion.fromVersion(it) }

        if (versionsInTag.isEmpty()) {
            throw IllegalStateException("No valid versioned files found in tag directory '$tag'")
        }

        // Вычисляем максимальную версию-таймстамп внутри этой папки
        val maxVersionInTag = versionsInTag.maxOrNull()
            ?: throw IllegalStateException("Could not determine max version for tag '$tag'")

        log.info { "Tag '$tag' successfully resolved to highest migration version: $maxVersionInTag" }

        // Запускаем откат до этой версии
        rollbackToVersion(maxVersionInTag.version)
    }

    /**
     * Вручную находит, вычитывает и применяет SQL файл отката, после чего чистит историю Flyway
     */
    private fun executeRollbackScript(version: String) {
        log.info { "Processing programmatic rollback for version: $version" }

        var sqlContent: String? = null
        var foundLocation: String? = null

        for (location in locations) {
            val cleanLocation = location.removePrefix("classpath:").removePrefix("/")
            val pattern = "classpath:$cleanLocation/$version"

            try {
                val resources = ResourcePatternUtils
                    .getResourcePatternResolver(resourceLoader)
                    .getResources("${pattern}_rb__*.sql")

                if (resources.isNotEmpty()) {
                    val resource = resources.first()
                    sqlContent = resource.inputStream.bufferedReader().use(BufferedReader::readText)
                    foundLocation = resource.filename
                    break
                }
            } catch (e: Exception) {
                // Игнорируем ошибки
            }
        }

        if (sqlContent.isNullOrBlank()) {
            throw IllegalStateException("Rollback script for version $version (matching *_rb__*.sql) was not found in locations: $locations")
        }

        log.info { "Found rollback script: $foundLocation. Executing SQL commands..." }

        jdbcTemplate.execute("set search_path to $activeSchema")
        jdbcTemplate.execute(sqlContent)

        val deletedRows = jdbcTemplate.update(
            "delete from $historyTable where version = ?",
            version
        )

        if (deletedRows == 0) {
            throw IllegalStateException("SQL executed, but failed to remove migration version $version from metadata table $historyTable")
        }

        log.info { "Metadata successfully synchronized. Version $version is now marked as unapplied." }
    }

    /**
     * Получает список строк-версий примененных миграций из БД, отсортированных по убыванию
     */
    private fun getAppliedMigrationsDesc(): List<String> {
        return jdbcTemplate.queryForList(
            "select version from $historyTable where success = true order by installed_rank desc",
            String::class.java
        ).filterNotNull()
    }
}
