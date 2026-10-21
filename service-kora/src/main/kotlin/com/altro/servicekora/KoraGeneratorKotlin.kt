package com.altro.servicekora

import java.io.*

class KoraGeneratorKotlin

fun main() {
    val TARGET_DIR = "./service-kora-benchmark-kotlin/src/main/kotlin"
    val RESOURCES_DIR = "./service-kora-benchmark-kotlin/src/main/resources"
    val totalNodes = 150

    val configsCount = maxOf(1, (totalNodes * 0.10).toInt())
    val controllersCount = maxOf(1, (totalNodes * 0.10).toInt())
    val reposCount = maxOf(1, (totalNodes * 0.13).toInt())
    val remainingNodes = totalNodes - (configsCount + controllersCount + reposCount)
    val servicesCount = remainingNodes / 2
    val mappersCount = remainingNodes - servicesCount

    val basePackageDir = File("$TARGET_DIR/io/koraframework/benchmark/kotlin")
    if (basePackageDir.exists()) {
        basePackageDir.deleteRecursively()
    }

    val migrationDir = File("$RESOURCES_DIR/db/migration")
    if (migrationDir.exists()) {
        migrationDir.deleteRecursively()
    }

    val paths = listOf("config", "api/dto", "repository", "service", "api")
    paths.forEach { File("$TARGET_DIR/io/koraframework/benchmark/kotlin/$it").mkdirs() }
    migrationDir.mkdirs()

    for (i in 0 until configsCount) {
        val file = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/config/SystemConfig$i.kt")
        file.writeText(
            """
            package io.koraframework.benchmark.kotlin.config

            import io.koraframework.config.kotlin.common.annotation.ConfigSource

            @ConfigSource("system.module$i")
            interface SystemConfig$i {
                fun environment(): String?
                fun version(): String?
            }
        """.trimIndent()
        )
    }

    for (i in 0 until mappersCount) {
        val dtoFile = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/api/dto/OrderDto$i.kt")
        dtoFile.writeText(
            """
            package io.koraframework.benchmark.kotlin.api.dto

            import io.koraframework.json.common.annotation.Json
            import java.math.BigDecimal

            @Json
            data class CreateOrderRequest$i(val userId: Long, val itemId: Long, val quantity: Int)

            @Json
            data class OrderResponse$i(val id: String, val userId: Long, val totalAmount: BigDecimal, val status: String)
        """.trimIndent()
        )

        val mapperFile = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/service/OrderMapper$i.kt")
        mapperFile.writeText(
            """
            package io.koraframework.benchmark.kotlin.service

            import io.koraframework.benchmark.kotlin.api.dto.OrderResponse$i
            import io.koraframework.common.annotation.Component
            import java.math.BigDecimal

            @Component
            open class OrderMapper$i {
                fun toResponse(id: String): OrderResponse$i {
                    return OrderResponse$i(id, 42L, BigDecimal.TEN, "GENERATED")
                }
            }
        """.trimIndent()
        )
    }

    val sqlBuilder = StringBuilder()

    for (i in 0 until reposCount) {
        val configId = i % configsCount
        val file = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/repository/OrderJdbcRepository$i.kt")

        file.writeText(
            """
            package io.koraframework.benchmark.kotlin.repository

            import io.koraframework.benchmark.kotlin.config.SystemConfig$configId
            import io.koraframework.database.common.annotation.Query
            import io.koraframework.database.common.annotation.Repository
            import io.koraframework.database.jdbc.JdbcRepository
            import java.math.BigDecimal

            @Repository
            interface OrderJdbcRepository$i : JdbcRepository {
                @Query("SELECT id FROM orders_$i WHERE id = :id")
                fun findById(id: String): String?

                @Query("INSERT INTO orders_$i(id, user_id, total_amount, status) VALUES (:id, 1, :totalAmount, 'PAID')")
                fun save(id: String, totalAmount: BigDecimal): String
            }
        """.trimIndent()
        )

        sqlBuilder.append(
            """
            CREATE TABLE orders_$i (
                id VARCHAR(50) PRIMARY KEY,
                user_id BIGINT NOT NULL,
                total_amount NUMERIC(19, 4) NOT NULL,
                status VARCHAR(50) NOT NULL
            );
            
            INSERT INTO orders_$i (id, user_id, total_amount, status)
            VALUES ('ORD-INIT-$i', 1, 15000.7500, 'PAID');
            
        """.trimIndent()
        ).append("\n")
    }

    val migrationFile = File(migrationDir, "V20261018131100__create_benchmark_orders_tables.sql")
    migrationFile.writeText(sqlBuilder.toString())
    // Продолжение функции main() из Части 1

    for (i in 0 until servicesCount) {
        val serviceReposCount = 1 + (i % 10)
        val serviceReposIds = (0 until serviceReposCount).map { (i + it) % reposCount }.distinct()
        val configId = i % configsCount

        val constructorFields = mutableListOf<String>()
        val imports = mutableListOf<String>()

        constructorFields.add("private val config: SystemConfig$configId")
        imports.add("import io.koraframework.benchmark.kotlin.config.SystemConfig$configId")

        serviceReposIds.forEach { id ->
            constructorFields.add("private val repository$id: OrderJdbcRepository$id")
            imports.add("import io.koraframework.benchmark.kotlin.repository.OrderJdbcRepository$id")
        }

        val childServicesCalls = mutableListOf<String>()
        if (i > 0) {
            val depServicesCount = 1 + (i % 2)
            val depServicesIds = (0 until depServicesCount).map { it * (i / depServicesCount) }.distinct()

            depServicesIds.forEach { depId ->
                if (depId < i) {
                    constructorFields.add("private val childService$depId: OrderService$depId")
                    childServicesCalls.add("childService$depId.getOrder(id)")
                }
            }
        }

        val repoCalls = serviceReposIds.map { id -> "repository$id.findById(id)" }
        val allCalls = (repoCalls + childServicesCalls).joinToString(" ?: ")
        val fieldsBlock = constructorFields.joinToString(",\n")

        val file = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/service/OrderService$i.kt")
        file.writeText(buildString {
            appendLine("package io.koraframework.benchmark.kotlin.service")
            appendLine()
            imports.distinct().forEach { appendLine(it) }
            appendLine("import io.koraframework.common.annotation.Component")
            appendLine("import java.util.UUID")
            appendLine("import java.math.BigDecimal")
            appendLine()
            appendLine("@Component")
            appendLine("open class OrderService$i(")
            fieldsBlock.lines().forEach { appendLine("    $it") }
            appendLine(") {")
            appendLine("    open fun getOrder(id: String): String? {")
            appendLine("        return $allCalls")
            appendLine("    }")
            appendLine()
            appendLine("    open fun createOrder(quantity: Int): String {")
            appendLine("        val id = UUID.randomUUID().toString()")
            appendLine("        repository${serviceReposIds.first()}.save(id, BigDecimal.valueOf(quantity.toLong()))")
            appendLine("        return id")
            appendLine("    }")
            appendLine("}")
        })
    }

    for (i in 0 until controllersCount) {
        val controllerServicesCount = 1 + (i % 2)
        val controllerServicesIds = (0 until controllerServicesCount).map { (i + it) % servicesCount }.distinct()
        val mapperId = i % mappersCount
        val configId = i % configsCount

        val constructorFields = mutableListOf<String>()
        constructorFields.add("private val config: SystemConfig$configId")

        controllerServicesIds.forEach { id ->
            constructorFields.add("private val orderService$id: OrderService$id")
        }
        constructorFields.add("private val orderMapper: OrderMapper$mapperId")

        val fieldsBlock = constructorFields.joinToString(",\n")
        val serviceCalls = controllerServicesIds.joinToString(" ?: ") { id -> "orderService$id.getOrder(id)" }

        val file = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/api/OrderController$i.kt")
        file.writeText(buildString {
            appendLine("package io.koraframework.benchmark.kotlin.api")
            appendLine()
            appendLine("import io.koraframework.benchmark.kotlin.api.dto.*")
            appendLine("import io.koraframework.benchmark.kotlin.config.SystemConfig$configId")
            controllerServicesIds.map { id -> "import io.koraframework.benchmark.kotlin.service.test.OrderService$id" }.distinct().forEach { appendLine(it) }
            appendLine("import io.koraframework.benchmark.kotlin.service.test.OrderMapper$mapperId")
            appendLine("import io.koraframework.common.annotation.Component")
            appendLine("import io.koraframework.http.common.HttpMethod")
            appendLine("import io.koraframework.http.common.annotation.HttpRoute")
            appendLine("import io.koraframework.http.common.annotation.Path")
            appendLine("import io.koraframework.http.server.common.annotation.HttpController")
            appendLine("import io.koraframework.json.common.annotation.Json")
            appendLine()
            appendLine("@HttpController")
            appendLine("@Component")
            appendLine("open class OrderController$i(")
            fieldsBlock.lines().forEach { appendLine("    $it") }
            appendLine(") {")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.GET, path = \"/orders/$i/{id}\")")
            appendLine("    open fun getOrder(@Path id: String): OrderResponse$i? {")
            appendLine("        val entity = $serviceCalls ?: throw RuntimeException(\"Not found\")")
            appendLine("        return orderMapper.toResponse(entity)")
            appendLine("    }")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.POST, path = \"/orders/$i\")")
            appendLine("    open fun checkout(@Json body: CreateOrderRequest$mapperId): OrderResponse$mapperId? {")
            appendLine("        val entity = orderService${controllerServicesIds.first()}.createOrder(body.quantity)")
            appendLine("        return orderMapper.toResponse(entity)")
            appendLine("    }")
            appendLine("}")
        })
    }

    val appFile = File("$TARGET_DIR/io/koraframework/benchmark/kotlin/App.kt")
    appFile.writeText(
        """
        package io.koraframework.benchmark.kotlin

        import io.koraframework.application.graph.*
        import io.koraframework.cache.caffeine.CaffeineCacheModule
        import io.koraframework.common.annotation.*
        import io.koraframework.config.hocon.*
        import io.koraframework.database.flyway.FlywayJdbcDatabaseModule
        import io.koraframework.database.jdbc.JdbcDatabaseModule
        import io.koraframework.http.server.undertow.*
        import io.koraframework.json.common.*
        import io.koraframework.logging.common.*
        import io.koraframework.logging.logback.*
        import io.koraframework.openapi.management.OpenApiManagementModule
        import io.koraframework.resilient.ResilientModule
        import io.koraframework.validation.module.ValidationModule

        @KoraApp
        interface App :
            HoconConfigModule,
            JsonModule,
            LoggingModule,
            LogbackModule,
            UndertowPublicHttpServerModule,
            JdbcDatabaseModule,
            FlywayJdbcDatabaseModule,
            CaffeineCacheModule,
            OpenApiManagementModule,
            ResilientModule,
            ValidationModule

        fun main() {
            KoraApplication.run(AppGraph::graph)
        }
    """.trimIndent()
    )


    val configContent = buildString {
        appendLine("app {")
        appendLine("  name = \"Task Management App\"")
        appendLine("  version = \"1.2.3\"")
        appendLine("  environment = \"development\"")
        appendLine("}")
        appendLine()
        appendLine("system {")
        appendLine("  version = \"1.2.3\"")
        appendLine("  environment = \"dev\"")

        val populatedConfigsCount = minOf(5, configsCount)
        for (i in 0 until populatedConfigsCount) {
            appendLine("  module$i {")
            appendLine("    environment = \"production\"")
            appendLine("    version = \"2.1.0\"")
            appendLine("  }")
        }
        appendLine("}")
        appendLine()
        appendLine(
            """
            httpServer {
              port = 8090
              telemetry {
                metrics { enabled = false }
                logging { enabled = false }
                tracing { enabled = false }
              }
            }

            jdbc {
              jdbcUrl = "jdbc:postgresql://localhost:5432/service_kora"
              schema = "custom"
              username = "service_kora"
              password = "1"
              poolName = "service_kora-pool"
            }

            flyway {
              enabled = true
              cleanDisabled = false
            }

            telemetry {
              "user.service.get" {
                logging { enabled = true, logLevel = "DEBUG" }
                metrics { enabled = true }
              }
            }

            resilient.retry.postgres-retry {
              delay = "100ms"
              attempts = 3
            }

            cache {
              users {
                expireAfterWrite = "10m"
                maximumSize = 1000
              }
            }

            openapi {
              management {
                enabled = true
                files = ["src/main/resources/openapi.yaml"]
              }
            }
        """.trimIndent()
        )
    }

    File(RESOURCES_DIR, "application.conf").writeText(configContent)
}
