package com.micronaut.bug.flyway.config

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.configuration.FluentConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayConfigurationCustomizer
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils

class FlywayRollbackConfigCustomizer(
    private val resourceLoader: ResourceLoader,
    private val flywayProperties: FlywayProperties,
) : FlywayConfigurationCustomizer {

    private val log = KotlinLogging.logger {}

    override fun customize(configuration: FluentConfiguration) {
        // --- ТОТАЛЬНАЯ ЗАЩИТА ОТ КРИТИЧЕСКИХ СВОЙСТВ ---
        configuration
            .group(false)        // Запрещаем объединение накатов в одну транзакцию
            .outOfOrder(false)   // Строго запрещаем нарушение хронологии версий
            .baselineOnMigrate(false) // Запрещаем базовые фейк-миграции

        val resolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)
        val rawLocation = flywayProperties.locations.firstOrNull() ?: "classpath:db/migration"
        val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/")

        val resources = resolver.getResources("classpath:$cleanBase/**/*.sql")

        // --- ВАЛИДАЦИЯ ПАРНОСТИ СТРУКТУРЫ (БЕЗ ДУБЛИРОВАНИЯ ЛОГОВ) ---
        val forwardMigrations = mutableMapOf<String, String>()
        val rollbackMigrations = mutableSetOf<String>()

        resources.forEach { resource ->
            val filename = resource.filename ?: return@forEach

            val versionMatch = Regex("""^\d+""").find(filename)
            if (versionMatch == null) {
                throw IllegalStateException("Flyway validation failed! File '$filename' does not start with a valid timestamp version.")
            }
            val version = versionMatch.value

            if (filename.contains("_rb__")) {
                rollbackMigrations.add(version)
            } else {
                if (forwardMigrations.containsKey(version)) {
                    throw IllegalStateException(
                        "Flyway validation failed! Duplicate version detected. " +
                                "Version '$version' is defined in multiple files: '${forwardMigrations[version]}' and '$filename'."
                    )
                }
                forwardMigrations[version] = filename
            }
        }

        forwardMigrations.forEach { (version, forwardFile) ->
            if (!rollbackMigrations.contains(version)) {
                throw IllegalStateException(
                    "Flyway validation failed! Missing rollback partner. " +
                            "Migration file '$forwardFile' exists, but no companion rollback file starting with '${version}_rb__' was found."
                )
            }
        }

        log.info { "Flyway fail-fast validation passed successfully. All migrations have valid '_rb__' partners." }

        // Динамически вычисляем подпапки версий релизов (X.X)
        // Используем деструктуризацию группы в регулярке, чтобы получить чисто "1.0", "2.0" вместо полного пути
        val releaseDirRegex = Regex("""$cleanBase/(\d+\.\d+)""")

        val activeLocations = resources
            .mapNotNull { resource ->
                val urlPath = resource.url.toString()
                val matchResult = releaseDirRegex.find(urlPath)
                // matchResult.groupValues[1] вытащит строго то, что попало в круглые скобки (\d+\.\d+)
                matchResult?.groupValues?.get(1)?.let { releaseVersion -> "classpath:$cleanBase/$releaseVersion" }
            }
            .distinct()
            .toTypedArray()

        if (activeLocations.isNotEmpty()) {
            configuration.locations(*activeLocations)
        } else {
            configuration.locations(rawLocation)
        }

        configuration
            .sqlMigrationPrefix("")
            .sqlMigrationSeparator("__")
    }
}
