package com.micronaut.bug.flyway

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

            // 1. Строго заглавная 'U' для Undo-скриптов
            if (filename.startsWith("U")) {
                val versionMatch = VERSION_REGEX.find(filename.substring(1)) ?: throw IllegalStateException(
                    "Undo migration file '$filename' does not contain a valid timestamp version after prefix."
                )
                undoMigrations.add(versionMatch.value)
                return@forEach
            }

            // 2. Строго заглавная 'V' для прямых миграций
            if (filename.startsWith("V")) {
                val versionMatch = VERSION_REGEX.find(filename.substring(1)) ?: throw IllegalStateException(
                    "Forward migration file '$filename' does not contain a valid timestamp version after prefix."
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

        // 3. Валидация парности V -> U (Проверка чистых таймстампов)
        forwardMigrations.forEach { (version, forwardFile) ->
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

        // 4. Динамический поиск подпапок с защитой от особенностей путей Windows/IDE/JAR
        val activeLocations = resources
            .mapNotNull { resource ->
                val cleanPath = try {
                    resource.file.absolutePath.replace(BACKSLASH, SLASH)
                } catch (e: Exception) {
                    val urlPath = resource.url.toString().replace(BACKSLASH, SLASH)
                    if (urlPath.contains("!")) urlPath.substringAfter("!") else urlPath
                }

                val tokens = cleanPath.split(SLASH).filter { it.isNotBlank() }

                // Ищем индекс нашей базовой папки (например, 'migration'), чтобы восстановить относительный путь
                val baseIndex = tokens.indexOfLast { it == cleanBase.substringAfterLast(SLASH) }
                if (baseIndex != -1 && tokens.size > baseIndex + 2) {
                    // Собираем весь хвост подпапок, идущих после db/migration, исключая сам файл
                    // Пример: из [..., db, migration, 1.x, 1.1, V1.sql] соберет "1.x/1.1"
                    val subDirs = tokens.subList(baseIndex + 1, tokens.size - 1).joinToString(SLASH)
                    "$CLASSPATH_PREFIX$cleanBase/$subDirs"
                } else {
                    null
                }
            }
            .distinct()
            .toTypedArray()

        if (activeLocations.isNotEmpty()) {
            configuration.locations(*activeLocations)
            log.info { "Flyway dynamic locations registered: ${activeLocations.joinToString()}" }
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

        private const val NO_ROLLBACK_SUFFIX = "_norb"
        private val VERSION_REGEX = Regex("""^\d+""")
    }
}
