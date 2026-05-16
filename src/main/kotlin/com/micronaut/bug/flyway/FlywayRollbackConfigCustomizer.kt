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
        // Защита от деструктивных настроек
        configuration
            .group(false)
            .outOfOrder(false)
            .baselineOnMigrate(false)

        val resolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)
        val rawLocation = flywayProperties.locations.firstOrNull() ?: DEFAULT_LOCATION
        val cleanBase = rawLocation.removePrefix(CLASSPATH_PREFIX).removePrefix(SLASH)

        val resources = resolver.getResources("$CLASSPATH_PREFIX$cleanBase/$SQL_ALL_PATTERN")

        val forwardMigrations = mutableMapOf<String, String>()
        val undoMigrations = mutableSetOf<String>()

        resources.forEach { resource ->
            val filename = resource.filename ?: return@forEach

            // 1. Собираем все наши самописные Undo-скрипты (начинаются с 'U')
            if (filename.startsWith("U")) {
                val versionMatch = VERSION_REGEX.find(filename.substring(1)) ?: throw IllegalStateException(
                    "Undo migration file '$filename' does not contain a valid timestamp version after 'U'."
                )
                undoMigrations.add(versionMatch.value)
                return@forEach
            }

            // 2. Собираем все прямые миграции (начинаются с 'V')
            if (filename.startsWith("V")) {
                val versionMatch = VERSION_REGEX.find(filename.substring(1)) ?: throw IllegalStateException(
                    "Forward migration file '$filename' does not contain a valid timestamp version after 'V'."
                )
                val version = versionMatch.value

                val existing = forwardMigrations.putIfAbsent(version, filename)
                if (existing != null) {
                    throw IllegalStateException(
                        "Flyway validation failed! Duplicate version detected. Version '$version' is defined in multiple files: '$existing' and '$filename'."
                    )
                }
            }
        }

        // 3. Валидация парности V -> U
        forwardMigrations.forEach { (version, forwardFile) ->
            // Если файл содержит суффикс _norb, ему не нужен U-скрипт отката
            if (forwardFile.contains(NO_ROLLBACK_SUFFIX)) {
                log.debug { "Migration '$forwardFile' is explicitly marked as non-rollable. Skipping U-twin validation." }
                return@forEach
            }

            if (!undoMigrations.contains(version)) {
                throw IllegalStateException(
                    "Flyway validation failed! Missing undo partner. Migration file '$forwardFile' exists, but no companion rollback file starting with 'U${version}__' was found. If rollback is impossible, add '$NO_ROLLBACK_SUFFIX' to the end of the filename (before .sql)."
                )
            }
        }

        log.info { "Flyway fail-fast validation passed successfully. Total forward migrations checked: ${forwardMigrations.size}" }

        // Динамический поиск подпапок (оставляем твою логику без изменений)
        val releaseDirRegex = Regex("""${Regex.escape(cleanBase)}/((.+/)?([^/]+))/[^/]+\.sql$""")
        val activeLocations = resources
            .mapNotNull { resource ->
                val urlPath = resource.url.toString().replace(BACKSLASH, SLASH)
                val matchResult = releaseDirRegex.find(urlPath)
                matchResult?.groupValues?.get(1)?.let { fullPath -> "$CLASSPATH_PREFIX$cleanBase/$fullPath" }
            }
            .distinct()
            .toTypedArray()

        if (activeLocations.isNotEmpty()) {
            configuration.locations(*activeLocations)
        } else {
            configuration.locations(rawLocation)
        }
    }

    companion object {
        private const val DEFAULT_LOCATION = "classpath:db/migration"
        private const val CLASSPATH_PREFIX = "classpath:"
        private const val SLASH = "/"
        private const val BACKSLASH = "\\"
        private const val SQL_ALL_PATTERN = "**/*.sql"

        // Теперь это суффикс в конце описания: пример V20260516140000__init_users_norb.sql
        private const val NO_ROLLBACK_SUFFIX = "_norb"
        private val VERSION_REGEX = Regex("""^\d+""")
    }
}
