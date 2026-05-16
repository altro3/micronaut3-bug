package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayRollbackDatabaseInitializer
import com.micronaut.bug.flyway.FlywayRollbackEngine
import com.micronaut.bug.flyway.FlywayRollbackStepExecutor
import org.flywaydb.core.Flyway
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.ApplicationContext
import org.springframework.context.annotation.Bean
import org.springframework.core.io.ResourceLoader
import org.springframework.jdbc.core.JdbcTemplate

@AutoConfiguration(before = [FlywayAutoConfiguration::class], after = [DataSourceAutoConfiguration::class])
@ConditionalOnClass(Flyway::class, JdbcTemplate::class)
@EnableConfigurationProperties(FlywayRollbackProperties::class, FlywayProperties::class)
class FlywayRollbackAutoConfig {

    @Bean
    fun flywayRollbackConfigCustomizer(
        resourceLoader: ResourceLoader,
        flywayProperties: FlywayProperties
    ) = FlywayRollbackConfigCustomizer(
        resourceLoader = resourceLoader,
        flywayProperties = flywayProperties,
    )

    @Bean
    fun flywayRollbackStepExecutor(jdbcTemplate: JdbcTemplate) =
        FlywayRollbackStepExecutor(jdbcTemplate)

    @Bean
    fun flywayRollbackEngine(
        flywayProperties: FlywayProperties,
        jdbcTemplate: JdbcTemplate,
        flywayRollbackStepExecutor: FlywayRollbackStepExecutor,
        resourceLoader: ResourceLoader
    ) = FlywayRollbackEngine(
        flywayProperties = flywayProperties,
        jdbcTemplate = jdbcTemplate,
        stepExecutor = flywayRollbackStepExecutor,
        resourceLoader = resourceLoader,
    )

    @Bean
    fun flywayRollbackDatabaseInitializer(
        rollbackEngine: FlywayRollbackEngine,
        properties: FlywayRollbackProperties,
        context: ApplicationContext
    ) = FlywayRollbackDatabaseInitializer(
        rollbackEngine = rollbackEngine,
        properties = properties,
        context = context,
    )
}

/*
давай теперь коротко сформулируем правила создания миграций и скриптов откатов к ним. Так же давай добавим правила именования миграций: директории - это версии релиза, и миграции
 */