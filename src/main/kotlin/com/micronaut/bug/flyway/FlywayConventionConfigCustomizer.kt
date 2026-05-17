package com.micronaut.bug.flyway

import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.api.configuration.FluentConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayConfigurationCustomizer

class FlywayConventionConfigCustomizer(
    private val metadataResolver: FlywayMetadataResolver,
    private val isRollbackEnabled: Boolean,
) : FlywayConfigurationCustomizer {

    private val log = KotlinLogging.logger {}

    override fun customize(configuration: FluentConfiguration) {
        configuration
            // 1. Управление шагами, линеаризацией и конкурентностью
            .group(CONFIG_GROUP)
            .outOfOrder(CONFIG_OUT_OF_ORDER)
            .baselineOnMigrate(CONFIG_BASELINE_ON_MIGRATE)

            // 2. Стабильность пула и Fail-Fast
            .connectRetries(CONFIG_CONNECT_RETRIES)
            .initSql(CONFIG_INIT_SQL)
            .failOnMissingLocations(CONFIG_FAIL_ON_MISSING_LOCATIONS)

            // 3. Строгий синтаксис, кодировки и принудительное отключение оверхеда
            .sqlMigrationSeparator(SEPARATOR_MIGRATION)
            .placeholderSeparator(SEPARATOR_PLACEHOLDER)
            .sqlMigrationPrefix(PREFIX_VERSIONED)
            .repeatableSqlMigrationPrefix(PREFIX_REPEATABLE)
            .placeholderPrefix(PREFIX_PLACEHOLDER)
            .scriptPlaceholderPrefix(PREFIX_SCRIPT_PLACEHOLDER)
            .encoding(Charsets.UTF_8)
            .ignoreMigrationPatterns(*CONFIG_IGNORE_PATTERNS)
            .detectEncoding(false)
            .skipDefaultCallbacks(true)
            .validateMigrationNaming(false)
            .validateOnMigrate(false) // Отключаем повторную стандартную валидацию Flyway
            .cleanDisabled(true)      // Железная защита продакшена от полного вайпа данных

        // Инициализируем и выставляем динамически отсканированные пути к папкам-листьям
        val allLocations = metadataResolver.getResolvedLocations()
        configuration.locations(*allLocations)
        log.info { "Flyway dynamic explicit locations registered: ${allLocations.joinToString()}" }

        if (!isRollbackEnabled) {
            metadataResolver.clearAllMetadata()
        }
    }

    companion object {
        private const val CONFIG_GROUP = false
        private const val CONFIG_OUT_OF_ORDER = false
        private const val CONFIG_BASELINE_ON_MIGRATE = false
        private const val CONFIG_CONNECT_RETRIES = 0
        private val CONFIG_INIT_SQL: String? = null
        private const val CONFIG_FAIL_ON_MISSING_LOCATIONS = true
        private val CONFIG_IGNORE_PATTERNS = arrayOf("*:*")
        private const val PREFIX_VERSIONED = "V"
        private const val PREFIX_UNDO = "U"
        private const val PREFIX_REPEATABLE = "R"
        private const val PREFIX_PLACEHOLDER = $$"${"
        private const val PREFIX_SCRIPT_PLACEHOLDER = $$"${"
        private const val SEPARATOR_MIGRATION = "__"
        private const val SEPARATOR_PLACEHOLDER = ":"
    }
}
