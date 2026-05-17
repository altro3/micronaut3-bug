package com.micronaut.bug.flyway

import com.micronaut.bug.flyway.config.FlywayRollbackProperties
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.NONE
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.STEPS
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.TAG
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.VERSION
import io.github.oshai.kotlinlogging.KotlinLogging
import org.flywaydb.core.Flyway
import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.flyway.FlywayMigrationStrategy
import org.springframework.context.ApplicationContext
import kotlin.system.exitProcess

class FlywayRollbackDatabaseInitializer(
    private val rollbackEngine: FlywayRollbackEngine,
    private val properties: FlywayRollbackProperties,
    private val context: ApplicationContext,
) : FlywayMigrationStrategy {

    private val log = KotlinLogging.logger {}

    override fun migrate(flyway: Flyway) {
        if (properties.mode != NONE) {
            log.info { "Rollback mode detected: ${properties.mode}. Executing rollback sequence BEFORE migration..." }

            try {
                when (properties.mode) {
                    STEPS -> {
                        log.info { "Rollback mode: STEPS. Target steps: ${properties.value}" }
                        rollbackEngine.rollback(properties.value.toInt())
                    }

                    VERSION -> {
                        log.info { "Rollback mode: VERSION. Target version: ${properties.value}" }
                        rollbackEngine.rollbackToVersion(properties.value)
                    }

                    TAG -> {
                        log.info { "Rollback mode: TAG. Target tag: ${properties.value}" }
                        rollbackEngine.rollbackToTag(properties.value)
                    }
                }
                shutdownApplication()
            } catch (e: Exception) {
                if (e is RollbackSuccessTestException) throw e

                log.error(e) { "CRITICAL: Rollback process failed! Aborting application startup to protect database integrity." }
                throw e
            }
        }

        log.info { "No rollback requested. Proceeding with standard Flyway migration..." }
        flyway.migrate()
    }

    private fun shutdownApplication() {
        log.info { "Database rollback completed successfully. Closing application context." }

        // Интеллектуальное определение тестового окружения по стектрейсу потока
        val isTestEnvironment = Thread.currentThread().stackTrace.any {
            it.className.contains("junit") || it.className.contains("cucumber") || it.className.contains("Test")
        }

        if (isTestEnvironment) {
            log.info { "Test environment detected. Bypassing hard exitProcess." }
            // Выбрасываем контролируемый маркер для остановки контекста Spring в тесте
            throw RollbackSuccessTestException("Rollback executed successfully in test mode.")
        }

        try {
            val exitCode = SpringApplication.exit(context, { 0 })
            exitProcess(exitCode)
        } catch (e: Exception) {
            log.error(e) { "Error during SpringApplication.exit, forcing hard JVM shutdown." }
            exitProcess(1)
        }
    }

    class RollbackSuccessTestException(message: String) : RuntimeException(message)
}
