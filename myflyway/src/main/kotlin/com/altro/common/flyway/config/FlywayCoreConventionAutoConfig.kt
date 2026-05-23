package com.altro.common.flyway.config

import com.altro.common.flyway.FlywayConventionConst.CONFIG_IGNORE_PATTERNS
import com.altro.common.flyway.FlywayConventionConst.PREFIX_PLACEHOLDER
import com.altro.common.flyway.FlywayConventionConst.PREFIX_REPEATABLE
import com.altro.common.flyway.FlywayConventionConst.PREFIX_SCRIPT_PLACEHOLDER
import com.altro.common.flyway.FlywayConventionConst.PREFIX_VERSIONED
import com.altro.common.flyway.FlywayConventionConst.SEPARATOR_MIGRATION
import com.altro.common.flyway.FlywayConventionConst.SEPARATOR_PLACEHOLDER
import com.altro.common.flyway.FlywayMetadataResolver
import org.flywaydb.core.Flyway
import org.flywaydb.core.api.Location
import org.flywaydb.core.api.configuration.ClassicConfiguration
import org.springframework.beans.factory.ObjectProvider
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnBooleanProperty
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayMigrationStrategy
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.core.io.ResourceLoader
import javax.sql.DataSource

@AutoConfiguration(
    before = [FlywayAutoConfiguration::class],
    after = [DataSourceAutoConfiguration::class]
)
@ConditionalOnBooleanProperty("spring.flyway.enabled", matchIfMissing = true)
@ConditionalOnClass(Flyway::class)
@EnableConfigurationProperties(FlywayProperties::class)
class FlywayCoreConventionAutoConfig {

    @Bean
    fun flywayMetadataResolver(
        resourceLoader: ResourceLoader,
        flywayProperties: FlywayProperties
    ) = FlywayMetadataResolver(resourceLoader, flywayProperties)

    @Bean
    @ConditionalOnMissingBean(Flyway::class)
    fun flyway(
        dataSource: DataSource,
        flywayProperties: FlywayProperties,
        flywayMetadataResolver: FlywayMetadataResolver,
        migrationStrategy: ObjectProvider<FlywayMigrationStrategy>
    ): Flyway {
        // Инициализируем ClassicConfiguration — единственный источник правды для ядра Flyway
        val configuration = ClassicConfiguration()
        configuration.dataSource = dataSource

        // =====================================================================
        // ЖЕСТКО ЗАПЕЧАТАННЫЕ КОНСТАНТЫ КОНВЕНЦИИ (ИХ НЕЛЬЗЯ ПЕРЕОПРЕДЕЛИТЬ)
        // =====================================================================
        configuration.isGroup = false
        configuration.isOutOfOrder = false
        configuration.isBaselineOnMigrate = false
        configuration.connectRetries = 0
        configuration.initSql = null
        configuration.isFailOnMissingLocations = true

        configuration.sqlMigrationSeparator = SEPARATOR_MIGRATION
        configuration.sqlMigrationPrefix = PREFIX_VERSIONED
        configuration.repeatableSqlMigrationPrefix = PREFIX_REPEATABLE
        configuration.placeholderPrefix = PREFIX_PLACEHOLDER
        configuration.scriptPlaceholderPrefix = PREFIX_SCRIPT_PLACEHOLDER
        configuration.placeholderSeparator = SEPARATOR_PLACEHOLDER
        configuration.setIgnoreMigrationPatterns(*CONFIG_IGNORE_PATTERNS)
        configuration.isDetectEncoding = false
        configuration.isSkipDefaultCallbacks = true
        configuration.isCleanDisabled = true
        configuration.isValidateOnMigrate = true
        configuration.isValidateMigrationNaming = false

        // Подхватываем только безопасные инфраструктурные проперти из yaml
        flywayProperties.table?.let { configuration.table = it }
        flywayProperties.tablespace?.let { configuration.tablespace = it }
        if (flywayProperties.schemas.isNotEmpty()) {
            configuration.setSchemas(flywayProperties.schemas.toTypedArray())
        }

        // Выставляем динамически отсканированные пути к папкам-листьям
        val allLocations = flywayMetadataResolver.getResolvedLocations()
        configuration.setLocations(*allLocations.map { Location(it) }.toTypedArray())

        // Собираем финальный инстанс Flyway
        val fluentConfiguration = Flyway.configure(configuration.classLoader)
        fluentConfiguration.configuration(configuration)
        val flyway = fluentConfiguration.load()

        // Передаем управление миграционной стратегии (роллбэк или обычный накат)
        val strategy = migrationStrategy.getIfAvailable()
        if (strategy != null) {
            strategy.migrate(flyway)
        } else {
            // Обычный старт (NONE) — перед накатом принудительно чистим RAM от кэша метаданных
            flywayMetadataResolver.clearAllMetadata()
            flyway.migrate()
        }

        return flyway
    }
}
