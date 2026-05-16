package com.micronaut.bug.flyway

import com.micronaut.bug.flyway.config.FlywayRollbackProperties
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.NONE
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.STEPS
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.TAG
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.VERSION
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.beans.factory.InitializingBean
import org.springframework.boot.SpringApplication
import org.springframework.context.ApplicationContext
import kotlin.system.exitProcess

class FlywayRollbackDatabaseInitializer(
    private val rollbackEngine: FlywayRollbackEngine,
    private val properties: FlywayRollbackProperties,
    private val context: ApplicationContext
) : InitializingBean {

    private val log = KotlinLogging.logger {}

    override fun afterPropertiesSet() {
        when (properties.mode) {
            NONE -> log.debug { "Flyway rollback mode is NONE. Proceeding with standard startup." }
            STEPS -> {
                log.info { "Rollback mode: STEPS. Target steps: ${properties.value}" }
                rollbackEngine.rollback(properties.value.toInt())
                shutdownApplication()
            }

            VERSION -> {
                log.info { "Rollback mode: VERSION. Target version: ${properties.value}" }
                rollbackEngine.rollbackToVersion(properties.value)
                shutdownApplication()
            }

            TAG -> {
                log.info { "Rollback mode: TAG. Target tag: ${properties.value}" }
                rollbackEngine.rollbackToTag(properties.value)
                shutdownApplication()
            }
        }
    }

    private fun shutdownApplication() {
        log.info { "Database rollback completed successfully. Closing application context." }
        // Элегантное завершение приложения Spring Boot с кодом 0
        val exitCode = SpringApplication.exit(context, { 0 })
        exitProcess(exitCode)
    }
}
