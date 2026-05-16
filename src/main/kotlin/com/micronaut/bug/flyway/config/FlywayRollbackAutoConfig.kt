package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayRollbackDatabaseInitializer
import com.micronaut.bug.flyway.FlywayRollbackEngine
import org.flywaydb.core.Flyway
import org.postgresql.Driver
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.AutoConfigureAfter
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Profile
import org.springframework.core.io.ResourceLoader
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.SimpleDriverDataSource
import org.springframework.util.ClassUtils
import javax.sql.DataSource

@AutoConfiguration(before = [FlywayAutoConfiguration::class], after = [DataSourceAutoConfiguration::class])
@ConditionalOnClass(Flyway::class, JdbcTemplate::class)
@EnableConfigurationProperties(
    FlywayRollbackProperties::class,
    FlywayProperties::class,
    DataSourceProperties::class,
)
class FlywayRollbackAutoConfig {

    @Bean
    @ConditionalOnMissingBean(DataSource::class)
    fun dataSource(dataSourceProperties: DataSourceProperties): DataSource {
        val isHikariPresent = ClassUtils.isPresent(
            "com.zaxxer.hikari.HikariDataSource",
            ClassUtils.getDefaultClassLoader()
        )

        return if (isHikariPresent) {
            dataSourceProperties.initializeDataSourceBuilder().build()
        } else {
            SimpleDriverDataSource().apply {
                driver = Driver()
                url = dataSourceProperties.determineUrl()
                username = dataSourceProperties.determineUsername()
                password = dataSourceProperties.determinePassword()
            }
        }
    }

    /**
     * 2. Гарантированно создаем JdbcTemplate, который нужен нашему движку
     */
    @Bean
    @ConditionalOnMissingBean(JdbcTemplate::class)
    fun jdbcTemplate(dataSource: DataSource): JdbcTemplate {
        return JdbcTemplate(dataSource)
    }

    /**
     * Самостоятельно собираем бин Flyway.
     * Теперь нам абсолютно все равно, почему стандартный FlywayAutoConfiguration не хотел запускаться.
     */
    @Bean
    @ConditionalOnMissingBean(Flyway::class)
    fun flyway(
        dataSource: DataSource,
        resourceLoader: ResourceLoader,
        flywayProperties: FlywayProperties
    ): Flyway {

        // Создаем наш кастомный конфигуратор путей и валидации парности _rb__
        val customizer = FlywayRollbackConfigCustomizer(resourceLoader, flywayProperties)

        // Конфигурируем базовый Flyway на основе DataSource
        val config = Flyway.configure().dataSource(dataSource)

        // Переносим стандартные настройки схем из application.yml (например, target)
        if (flywayProperties.schemas.isNotEmpty()) {
            config.schemas(*flywayProperties.schemas.toTypedArray())
        }
        if (flywayProperties.table.isNotBlank()) {
            config.table(flywayProperties.table)
        }

        // Запускаем кастомайзер: он проверит парность файлов и пропишет точные папки релизов 1.0, 2.0
        customizer.customize(config)

        val flywayInstance = config.load()

        // Выполняем стандартный автоматический накат миграций при старте
        flywayInstance.migrate()

        return flywayInstance
    }

    /**
     * ФАЗА 2: Выполняется СТРОГО ПОСЛЕ автоконфигурации Flyway.
     * Когда бин Flyway уже гарантированно создан, мы спокойно собираем наш движок и инициализатор.
     */
    @Configuration(proxyBeanMethods = false)
    @AutoConfigureAfter(FlywayAutoConfiguration::class)
    class EngineConfiguration {
        @Bean
        fun flywayRollbackEngine(
            flyway: Flyway,
            jdbcTemplate: JdbcTemplate,
            resourceLoader: ResourceLoader,
        ) = FlywayRollbackEngine(
            flyway = flyway,
            jdbcTemplate = jdbcTemplate,
            resourceLoader = resourceLoader,
            locations = flyway.configuration.locations.map { it.descriptor }
        )

        @Bean
        @Profile("!test")
        fun flywayRollbackDatabaseInitializer(
            flywayRollbackEngine: FlywayRollbackEngine,
            properties: FlywayRollbackProperties
        ) = FlywayRollbackDatabaseInitializer(
            rollbackEngine = flywayRollbackEngine,
            properties = properties,
        )
    }
}
