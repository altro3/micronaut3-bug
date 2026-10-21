package com.altro.servicekora

import java.io.*

class KoraGeneratorJava

fun main() {
    val TARGET_DIR = "./service-kora-benchmark-java/src/main/java"
    val RESOURCES_DIR = "./service-kora-benchmark-java/src/main/resources"
    val totalNodes = 15

    val configsCount = maxOf(1, (totalNodes * 0.10).toInt())
    val controllersCount = maxOf(1, (totalNodes * 0.10).toInt())
    val reposCount = maxOf(1, (totalNodes * 0.13).toInt())
    val remainingNodes = totalNodes - (configsCount + controllersCount + reposCount)
    val servicesCount = remainingNodes / 2
    val mappersCount = remainingNodes - servicesCount

    val basePackageDir = File("$TARGET_DIR/io/koraframework/benchmark/java")
    if (basePackageDir.exists()) {
        basePackageDir.deleteRecursively()
    }

    val migrationDir = File("$RESOURCES_DIR/db/migration")
    if (migrationDir.exists()) {
        migrationDir.deleteRecursively()
    }

    val paths = listOf("config", "api/dto", "repository", "service", "api")
    paths.forEach { File("$TARGET_DIR/io/koraframework/benchmark/java/$it").mkdirs() }
    migrationDir.mkdirs()

    for (i in 0 until configsCount) {
        val file = File("$TARGET_DIR/io/koraframework/benchmark/java/config/SystemConfig$i.java")
        file.writeText(
            """
            package io.koraframework.benchmark.java.config;

            import io.koraframework.config.common.annotation.ConfigSource;

            @ConfigSource("system.module$i")
            public interface SystemConfig$i {
                String environment();
                String version();
            }
            """.trimIndent()
        )
    }

    for (i in 0 until mappersCount) {
        val dtoFile = File("$TARGET_DIR/io/koraframework/benchmark/java/api/dto/OrderDto$i.java")
        dtoFile.writeText(
            """
            package io.koraframework.benchmark.java.api.dto;

            import io.koraframework.json.common.annotation.Json;
            import java.math.BigDecimal;

            public final class OrderDto$i {
                @Json
                public record CreateOrderRequest$i(Long userId, Long itemId, Integer quantity) {}

                @Json
                public record OrderResponse$i(String id, Long userId, BigDecimal totalAmount, String status) {}
            }
            """.trimIndent()
        )

        val mapperFile = File("$TARGET_DIR/io/koraframework/benchmark/java/service/OrderMapper$i.java")
        mapperFile.writeText(
            """
            package io.koraframework.benchmark.java.service;

            import io.koraframework.benchmark.java.api.dto.OrderDto$i.OrderResponse$i;
            import io.koraframework.common.annotation.Component;
            import java.math.BigDecimal;

            @Component
            public class OrderMapper$i {
                public OrderResponse$i toResponse(String id) {
                    return new OrderResponse$i(id, 42L, BigDecimal.TEN, "GENERATED");
                }
            }
            """.trimIndent()
        )
    }

    val sqlBuilder = StringBuilder()

    for (i in 0 until reposCount) {
        val configId = i % configsCount
        val file = File("$TARGET_DIR/io/koraframework/benchmark/java/repository/OrderJdbcRepository$i.java")

        file.writeText(
            """
            package io.koraframework.benchmark.java.repository;

            import io.koraframework.database.common.annotation.Query;
            import io.koraframework.database.common.annotation.Repository;
            import io.koraframework.database.jdbc.JdbcRepository;
            import java.math.BigDecimal;

            @Repository
            public interface OrderJdbcRepository$i extends JdbcRepository {
                @Query("SELECT id FROM orders_$i WHERE id = :id")
                String findById(String id);

                @Query("INSERT INTO orders_$i(id, user_id, total_amount, status) VALUES (:id, 1, :totalAmount, 'PAID')")
                String save(String id, BigDecimal totalAmount);
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
    for (i in 0 until servicesCount) {
        val serviceReposCount = 1 + (i % 10)
        val serviceReposIds = (0 until serviceReposCount).map { (i + it) % reposCount }.distinct()
        val configId = i % configsCount

        val fields = mutableListOf<String>()
        val constructorParams = mutableListOf<String>()
        val constructorAssignments = mutableListOf<String>()
        val imports = mutableListOf<String>()

        fields.add("private final SystemConfig$configId config;")
        constructorParams.add("SystemConfig$configId config")
        constructorAssignments.add("this.config = config;")
        imports.add("import io.koraframework.benchmark.java.config.SystemConfig$configId;")

        serviceReposIds.forEach { id ->
            fields.add("private final OrderJdbcRepository$id repository$id;")
            constructorParams.add("OrderJdbcRepository$id repository$id")
            constructorAssignments.add("this.repository$id = repository$id;")
            imports.add("import io.koraframework.benchmark.java.repository.OrderJdbcRepository$id;")
        }

        val childServicesCalls = mutableListOf<String>()
        if (i > 0) {
            val depServicesCount = 1 + (i % 2)
            val depServicesIds = (0 until depServicesCount).map { it * (i / depServicesCount) }.distinct()

            depServicesIds.forEach { depId ->
                if (depId < i) {
                    fields.add("private final OrderService$depId childService$depId;")
                    constructorParams.add("OrderService$depId childService$depId")
                    constructorAssignments.add("this.childService$depId = childService$depId;")
                    childServicesCalls.add("this.childService$depId.getOrder(id)")
                }
            }
        }

        val elvisChain = buildString {
            serviceReposIds.forEachIndexed { idx, id ->
                appendLine("        String repoRes$idx = this.repository$id.findById(id);")
                appendLine("        if (repoRes$idx != null) return repoRes$idx;")
            }
            childServicesCalls.forEachIndexed { idx, call ->
                appendLine("        String childRes$idx = $call;")
                appendLine("        if (childRes$idx != null) return childRes$idx;")
            }
            append("        return null;")
        }

        val file = File("$TARGET_DIR/io/koraframework/benchmark/java/service/OrderService$i.java")
        file.writeText(buildString {
            appendLine("package io.koraframework.benchmark.java.service;")
            appendLine()
            imports.distinct().sorted().forEach { appendLine(it) }
            appendLine("import io.koraframework.common.annotation.Component;")
            appendLine("import java.util.UUID;")
            appendLine("import java.math.BigDecimal;")
            appendLine()
            appendLine("@Component")
            appendLine("public class OrderService$i {")
            fields.forEach { appendLine("    $it") }
            appendLine()
            appendLine("    public OrderService$i(")
            appendLine("        " + constructorParams.joinToString(",\n        "))
            appendLine("    ) {")
            constructorAssignments.forEach { appendLine("        $it") }
            appendLine("    }")
            appendLine()
            appendLine("    public String getOrder(String id) {")
            appendLine(elvisChain)
            appendLine("    }")
            appendLine()
            appendLine("    public String createOrder(int quantity) {")
            appendLine("        String id = UUID.randomUUID().toString();")
            appendLine("        this.repository${serviceReposIds.first()}.save(id, BigDecimal.valueOf(quantity));")
            appendLine("        return id;")
            appendLine("    }")
            appendLine("}")
        })
    }
    for (i in 0 until controllersCount) {
        val controllerServicesCount = 1 + (i % 2)
        val controllerServicesIds = (0 until controllerServicesCount).map { (i + it) % servicesCount }.distinct()
        val mapperId = i % mappersCount
        val configId = i % configsCount

        val fields = mutableListOf<String>()
        val constructorParams = mutableListOf<String>()
        val constructorAssignments = mutableListOf<String>()

        fields.add("private final SystemConfig$configId config;")
        constructorParams.add("SystemConfig$configId config")
        constructorAssignments.add("this.config = config;")

        controllerServicesIds.forEach { id ->
            fields.add("private final OrderService$id orderService$id;")
            constructorParams.add("OrderService$id orderService$id")
            constructorAssignments.add("this.orderService$id = orderService$id;")
        }
        fields.add("private final OrderMapper$mapperId orderMapper;")
        constructorParams.add("OrderMapper$mapperId orderMapper")
        constructorAssignments.add("this.orderMapper = orderMapper;")

        val controllerElvisChain = buildString {
            controllerServicesIds.forEachIndexed { idx, id ->
                appendLine("        String sRes$idx = this.orderService$id.getOrder(id);")
                appendLine("        if (sRes$idx != null) entity = sRes$idx;")
            }
        }

        val file = File("$TARGET_DIR/io/koraframework/benchmark/java/api/OrderController$i.java")
        file.writeText(buildString {
            appendLine("package io.koraframework.benchmark.java.api;")
            appendLine()
            appendLine("import io.koraframework.benchmark.java.api.dto.OrderDto$mapperId.*;")
            appendLine("import io.koraframework.benchmark.java.config.SystemConfig$configId;")
            controllerServicesIds.map { id -> "import io.koraframework.benchmark.java.service.OrderService$id;" }.distinct().forEach { appendLine(it) }
            appendLine("import io.koraframework.benchmark.java.service.OrderMapper$mapperId;")
            appendLine("import io.koraframework.common.annotation.Component;")
            appendLine("import io.koraframework.http.common.HttpMethod;")
            appendLine("import io.koraframework.http.common.annotation.HttpRoute;")
            appendLine("import io.koraframework.http.common.annotation.Path;")
            appendLine("import io.koraframework.http.server.common.annotation.HttpController;")
            appendLine("import io.koraframework.json.common.annotation.Json;")
            appendLine()
            appendLine("@HttpController")
            appendLine("@Component")
            appendLine("public class OrderController$i {")
            appendLine()
            fields.forEach { appendLine("    $it") }
            appendLine()
            appendLine("    public OrderController$i(")
            appendLine("        " + constructorParams.joinToString(",\n        "))
            appendLine("    ) {")
            constructorAssignments.forEach { appendLine("        $it") }
            appendLine("    }")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.GET, path = \"/orders/$i/{id}\")")
            appendLine("    public OrderResponse$mapperId getOrder(@Path String id) {")
            appendLine("        String entity = null;")
            appendLine(controllerElvisChain)
            appendLine("        if (entity == null) throw new RuntimeException(\"Not found\");")
            appendLine("        return this.orderMapper.toResponse(entity);")
            appendLine("    }")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.POST, path = \"/orders/$i\")")
            appendLine("    public OrderResponse$mapperId checkout(@Json CreateOrderRequest$mapperId body) {")
            appendLine("        String entity = this.orderService${controllerServicesIds.first()}.createOrder(body.quantity());")
            appendLine("        return this.orderMapper.toResponse(entity);")
            appendLine("    }")
            appendLine("}")
        })
    }

    val appFile = File("$TARGET_DIR/io/koraframework/benchmark/java/App.java")
    appFile.writeText(
        """
        package io.koraframework.benchmark.java;

        import io.koraframework.application.graph.KoraApplication;
        import io.koraframework.common.annotation.KoraApp;
        import io.koraframework.config.hocon.HoconConfigModule;
        import io.koraframework.json.common.JsonModule;
        import io.koraframework.logging.common.LoggingModule;
        import io.koraframework.logging.logback.LogbackModule;
        import io.koraframework.http.server.undertow.UndertowPublicHttpServerModule;
        import io.koraframework.database.jdbc.JdbcDatabaseModule;
        import io.koraframework.database.flyway.FlywayJdbcDatabaseModule;
        import io.koraframework.cache.caffeine.CaffeineCacheModule;
        import io.koraframework.openapi.management.OpenApiManagementModule;
        import io.koraframework.resilient.ResilientModule;
        import io.koraframework.validation.module.ValidationModule;

        @KoraApp
        public interface App extends
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
            ValidationModule {

            static void main(String[] args) {
                KoraApplication.run(AppGraph::graph);
            }
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
