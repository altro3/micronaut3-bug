package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayMetadataResolver
import com.micronaut.bug.flyway.FlywayRollbackDatabaseInitializer
import com.micronaut.bug.flyway.FlywayRollbackEngine
import com.micronaut.bug.flyway.FlywayRollbackStepExecutor
import com.micronaut.bug.flyway.config.FlywayRollbackAutoConfig.OnRollbackEnabledCondition
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.NONE
import org.flywaydb.core.Flyway
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionMessage
import org.springframework.boot.autoconfigure.condition.ConditionOutcome
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.condition.SpringBootCondition
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayMigrationStrategy
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.ApplicationContext
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.ConditionContext
import org.springframework.context.annotation.Conditional
import org.springframework.core.io.ResourceLoader
import org.springframework.core.type.AnnotatedTypeMetadata
import org.springframework.jdbc.core.JdbcTemplate
import javax.sql.DataSource

@AutoConfiguration(before = [FlywayAutoConfiguration::class], after = [DataSourceAutoConfiguration::class])
@ConditionalOnClass(Flyway::class, JdbcTemplate::class)
@Conditional(OnRollbackEnabledCondition::class)
@EnableConfigurationProperties(FlywayRollbackProperties::class, FlywayProperties::class)
class FlywayRollbackAutoConfig {

    @Bean
    fun flywayRollbackStepExecutor(dataSource: DataSource) =
        FlywayRollbackStepExecutor(JdbcTemplate(dataSource))

    @Bean
    fun flywayRollbackEngine(
        flywayMetadataResolver: FlywayMetadataResolver,
        flywayProperties: FlywayProperties,
        flywayRollbackProperties: FlywayRollbackProperties,
        dataSource: DataSource,
        flywayRollbackStepExecutor: FlywayRollbackStepExecutor,
    ) = FlywayRollbackEngine(
        flywayProperties = flywayProperties,
        flywayRollbackProperties = flywayRollbackProperties,
        jdbcTemplate = JdbcTemplate(dataSource),
        stepExecutor = flywayRollbackStepExecutor,
        metadataResolver = flywayMetadataResolver,
    )

    @Bean
    fun flywayRollbackDatabaseInitializer(
        rollbackEngine: FlywayRollbackEngine,
        properties: FlywayRollbackProperties,
        context: ApplicationContext,
    ): FlywayMigrationStrategy = FlywayRollbackDatabaseInitializer(
        rollbackEngine = rollbackEngine,
        properties = properties,
        context = context,
    )

    class OnRollbackEnabledCondition : SpringBootCondition() {
        override fun getMatchOutcome(context: ConditionContext, metadata: AnnotatedTypeMetadata): ConditionOutcome {
            val mode = context.environment.getProperty("spring.flyway.rollback.mode")

            if (mode.isNullOrBlank() || mode.equals(NONE.name, ignoreCase = true)) {
                return ConditionOutcome.noMatch(
                    ConditionMessage.forCondition("FlywayRollbackMode")
                        .because("spring.flyway.rollback.mode is missing or set to NONE")
                )
            }

            return ConditionOutcome.match(
                ConditionMessage.forCondition("FlywayRollbackMode")
                    .because("spring.flyway.rollback.mode is set to $mode")
            )
        }
    }
}

/*
давай теперь коротко сформулируем правила создания миграций и скриптов откатов к ним. Так же давай добавим правила именования миграций: директории - это версии релиза, и миграции
 */