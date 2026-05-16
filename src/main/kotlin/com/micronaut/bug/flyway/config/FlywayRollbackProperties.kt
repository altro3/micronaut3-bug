package com.micronaut.bug.flyway.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties("spring.flyway.rollback")
data class FlywayRollbackProperties(
    /**
     * Режим отката. Возможные значения: NONE, STEPS, VERSION, TAG.
     */
    val mode: RollbackMode = RollbackMode.NONE,
    /**
     * Количество шагов для отката (используется при mode = STEPS).
     */
    val steps: Int = 1,
    /**
     * Целевая версия для отката (используется при mode = VERSION).
     */
    val version: String? = null,
    /**
     * Целевой тег/маркер в описании миграции для отката (используется при mode = TAG).
     */
    val tag: String? = null
) {
    enum class RollbackMode {
        NONE,
        STEPS,
        VERSION,
        TAG,
    }
}
