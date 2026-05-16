package com.micronaut.bug.flyway.config

import com.micronaut.bug.flyway.FlywayRollbackDatabaseInitializer
import com.micronaut.bug.flyway.FlywayRollbackEngine
import org.flywaydb.core.Flyway
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration
import org.springframework.boot.autoconfigure.flyway.FlywayConfigurationCustomizer
import org.springframework.boot.autoconfigure.flyway.FlywayProperties
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Profile
import org.springframework.core.io.ResourceLoader
import org.springframework.core.io.support.ResourcePatternUtils
import org.springframework.jdbc.core.JdbcTemplate

@AutoConfiguration(after = [FlywayAutoConfiguration::class])
@ConditionalOnClass(Flyway::class, JdbcTemplate::class)
@EnableConfigurationProperties(FlywayRollbackProperties::class)
class FlywayRollbackAutoConfig {

    @Bean
    fun flywayConfigurationCustomizer(
        resourceLoader: ResourceLoader,
        flywayProperties: FlywayProperties // Внедряем стандартные свойства Spring Boot для Flyway
    ): FlywayConfigurationCustomizer {
        return FlywayConfigurationCustomizer { configuration ->
            val resolver = ResourcePatternUtils.getResourcePatternResolver(resourceLoader)

            // Берём первый настроенный путь из стандартного spring.flyway.locations (по дефолту там ["classpath:db/migration"])
            val rawLocation = flywayProperties.locations.firstOrNull() ?: "classpath:db/migration"
            val cleanBase = rawLocation.removePrefix("classpath:").removePrefix("/")

            // Сканируем этот стандартный корень на наличие SQL-файлов
            val resources = resolver.getResources("classpath:$cleanBase/**/*.sql")

            // Динамически вычисляем подпапки релизов (X.X) внутри стандартного корня
            val activeLocations = resources
                .mapNotNull { resource ->
                    val urlPath = resource.url.toString()
                    val match = Regex("""$cleanBase/\d+\.\d+""").find(urlPath)
                    match?.value?.let { "classpath:$it" }
                }
                .distinct()
                .toTypedArray()

            // Прокидываем вычисленные точные папки во Flyway
            if (activeLocations.isNotEmpty()) {
                configuration.locations(*activeLocations)
            } else {
                configuration.locations(rawLocation)
            }

            configuration
                .sqlMigrationPrefix("")
                .sqlMigrationSeparator("__")
        }
    }

    @Bean
    @ConditionalOnBean(Flyway::class)
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
