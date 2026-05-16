package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.Flyway
import org.flywaydb.core.api.MigrationVersion
import org.springframework.core.io.ResourceLoader
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
    private val historyTable = flyway.configuration.table

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
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    open fun rollbackToTag(tag: String) {
        log.info { "Starting Community-style programmatic rollback to tag: $tag" }

        val info = flyway.info()
        val tagMigration = info.applied()
            .filter { it.version != null }
            .firstOrNull { it.description.contains(tag, ignoreCase = true) }

        if (tagMigration == null) {
            throw IllegalArgumentException("Tag '$tag' matching any applied migration description was not found.")
        }

        log.info { "Found target tag '$tag' at version: ${tagMigration.version}" }
        rollbackToVersion(tagMigration.version.version)
    }

    /**
     * Вручную находит, вычитывает и применяет SQL файл отката, после чего чистит историю Flyway
     */
    private fun executeRollbackScript(version: String) {
        log.info { "Processing programmatic rollback for version: $version" }

        // 1. Пытаемся найти файл отката во всех зарегистрированных каталогах (1.0, 2.0, 3.0 и т.д.)
        var sqlContent: String? = null
        var foundLocation: String? = null

        for (location in locations) {
            // Формируем маску поиска: ищем файл, начинающийся с версии и содержащий маркер отката "_rb__"
            val cleanLocation = location.removePrefix("classpath:")
            val pattern = "classpath:$cleanLocation/$version"

            try {
                // Пытаемся найти файл, сканируя ресурсы через ResourceLoader Спринга.
                // Так как точное имя описания скрипта мы не знаем, мы ищем файл по совпадению префикса версии.
                // Для простоты и отказоустойчивости, ищем напрямую файл отката, зная структуру имен.
                val resources = org.springframework.core.io.support.ResourcePatternUtils
                    .getResourcePatternResolver(resourceLoader)
                    .getResources("${pattern}_rb__*.sql")

                if (resources.isNotEmpty()) {
                    val resource = resources.first()
                    sqlContent = resource.inputStream.bufferedReader().use(BufferedReader::readText)
                    foundLocation = resource.filename
                    break
                }
            } catch (e: Exception) {
                // Игнорируем ошибки сканирования конкретной папки, ищем в следующей
            }
        }

        if (sqlContent.isNullOrBlank()) {
            throw IllegalStateException("Rollback script for version $version (matching *_rb__*.sql) was not found in locations: $locations")
        }

        log.info { "Found rollback script: $foundLocation. Executing SQL commands..." }

        // 2. Выполняем тело скрипта отката в базе данных
        jdbcTemplate.execute(sqlContent)

        // 3. Вручную вырезаем запись из таблицы истории Flyway, чтобы восстановить синхронизацию схемы
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
