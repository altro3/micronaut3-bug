package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.NONE
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.STEPS
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.TAG
import com.micronaut.bug.flyway.config.FlywayRollbackProperties.RollbackMode.VERSION
import org.flywaydb.core.api.MigrationVersion
import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@ConfigurationProperties("spring.flyway.rollback")
class FlywayRollbackProperties(
    val mode: RollbackMode = NONE,
    val value: String = "",
    val force: Boolean = false,
) {

    init {
        when (mode) {
            NONE -> {
                // Для режима NONE значение value может быть любым (в том числе пустым)
            }

            STEPS -> {
                val steps = value.toIntOrNull()
                    ?: throw IllegalArgumentException("Steps must be positive integer: '$value'")
                if (steps <= 0) {
                    throw IllegalArgumentException("Steps must be positive integer: '$value'")
                }
            }

            VERSION -> {
                if (value.isBlank()) {
                    throw IllegalArgumentException("Rollback value must not be blank when mode is VERSION")
                }

                if (value.length != 14) {
                    throw IllegalArgumentException("Invalid version length: '$value'. Expected exactly 14 characters for format YYYYMMDDHHMMSS.")
                }

                try {
                    val formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss")
                    LocalDateTime.parse(value, formatter)
                } catch (e: Exception) {
                    throw IllegalArgumentException("Invalid version timestamp: '$value'. It must be a valid calendar date in format YYYYMMDDHHMMSS.", e)
                }
            }

            TAG -> {
                if (value.isBlank()) {
                    throw IllegalArgumentException("Rollback value must not be blank when mode is TAG")
                }
                try {
                    MigrationVersion.fromVersion(value)
                } catch (e: Exception) {
                    throw IllegalArgumentException("Invalid TAG format: '$value'. Tag directory name must be a valid version format (e.g., '2.0' or '1.1.5').", e)
                }
            }
        }
    }

    enum class RollbackMode {
        NONE,
        STEPS,
        VERSION,
        TAG,
    }
}
