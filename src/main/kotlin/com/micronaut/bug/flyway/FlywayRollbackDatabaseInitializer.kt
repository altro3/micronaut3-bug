package com.micronaut.bug.flyway

import com.micronaut.bug.flyway.config.FlywayRollbackProperties
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.NONE
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.STEPS
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.TAG
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.VERSION
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.beans.factory.InitializingBean
import kotlin.system.exitProcess

class FlywayRollbackDatabaseInitializer(
    private val rollbackEngine: FlywayRollbackEngine,
    private val properties: FlywayRollbackProperties
) : InitializingBean {

    private val log = KotlinLogging.logger {}

    override fun afterPropertiesSet() {
        when (properties.mode) {
            NONE -> {
                // Ничего не делаем, приложение продолжит стандартный запуск и накат миграций
                log.debug { "Flyway rollback mode is NONE. Proceeding with standard startup." }
            }

            STEPS -> {
                log.info { "Rollback mode DETECTED: STEPS. Target steps: ${properties.steps}" }
                rollbackEngine.rollback(properties.steps)
                shutdownApplication()
            }

            VERSION -> {
                val targetVersion = properties.version
                require(!targetVersion.isNullOrBlank()) { "Rollback mode is VERSION, but 'spring.flyway.rollback.version' is missing!" }
                log.info { "Rollback mode DETECTED: VERSION. Target version: $targetVersion" }
                rollbackEngine.rollbackToVersion(targetVersion)
                shutdownApplication()
            }

            TAG -> {
                val targetTag = properties.tag
                require(!targetTag.isNullOrBlank()) { "Rollback mode is TAG, but 'spring.flyway.rollback.tag' is missing!" }
                log.info { "Rollback mode DETECTED: TAG. Target tag: $targetTag" }
                rollbackEngine.rollbackToTag(targetTag)
                shutdownApplication()
            }
        }
    }

    private fun shutdownApplication() {
        log.info { "Database rollback completed successfully. Terminating application process." }
        // Выходим с кодом 0, чтобы Kubernetes / CI-пайплайн понял, что задача выполнена успешно
        exitProcess(0)
    }
}
