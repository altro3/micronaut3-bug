package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayConventionConfigCustomizer
import com.micronaut.bug.flyway.FlywayMetadataResolver
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode
import org.flywaydb.core.Flyway
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.core.io.ResourceLoader

@AutoConfiguration(before = [FlywayAutoConfiguration::class], after = [DataSourceAutoConfiguration::class])
@ConditionalOnClass(Flyway::class)
@EnableConfigurationProperties(FlywayProperties::class)
class FlywayCoreConventionAutoConfig {

    @Bean
    fun flywayMetadataResolver(
        resourceLoader: ResourceLoader,
        flywayProperties: FlywayProperties
    ) = FlywayMetadataResolver(resourceLoader, flywayProperties)

    @Bean
    fun flywayConventionConfigCustomizer(
        flywayMetadataResolver: FlywayMetadataResolver,
        @Value($$"${spring.flyway.rollback.mode:NONE}")
        mode: RollbackMode,
    ) = FlywayConventionConfigCustomizer(
        metadataResolver = flywayMetadataResolver,
        isRollbackEnabled = mode != RollbackMode.NONE,
    )
}
