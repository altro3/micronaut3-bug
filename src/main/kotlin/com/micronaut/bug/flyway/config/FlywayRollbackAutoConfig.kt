package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayRollbackDatabaseInitializer
import com.micronaut.bug.flyway.FlywayRollbackEngine
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
        flywayProperties = flywayProperties
    )

    @Bean
    fun flywayRollbackEngine(
        flyway: Flyway,
        jdbcTemplate: JdbcTemplate,
        resourceLoader: ResourceLoader
    ) = FlywayRollbackEngine(
        flyway = flyway,
        jdbcTemplate = jdbcTemplate,
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
