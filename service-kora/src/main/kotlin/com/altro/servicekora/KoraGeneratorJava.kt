package com.altro.servicekora

import java.io.*

class KoraGeneratorJava

fun main() {
    val TARGET_DIR = "./service-kora-benchmark-java/src/main/java"
    val RESOURCES_DIR = "./service-kora-benchmark-java/src/main/resources"

    val includeConfigs = true
    val includeRepos = true
    val includeMappers = true
    val includeServices = true

    val configsPart = 0.10
    val reposPart = 0.13
    val mappersPart = 0.12
    val controllersPart = 0.10

    val totalNodes = 10

    val configsCount = if (includeConfigs) maxOf(1, (totalNodes * configsPart).toInt()) else 0
    val controllersCount = if (includeServices) maxOf(1, (totalNodes * controllersPart).toInt()) else 0
    val reposCount = if (includeRepos) maxOf(1, (totalNodes * reposPart).toInt()) else 0
    val mappersCount = if (includeMappers) maxOf(1, (totalNodes * mappersPart).toInt()) else 0
    val servicesCount = maxOf(1, totalNodes - configsCount - reposCount - controllersCount - mappersCount)

    val basePackageDir = File("$TARGET_DIR/io/koraframework/benchmark/java")
    if (basePackageDir.exists()) basePackageDir.deleteRecursively()

    val migrationDir = File("$RESOURCES_DIR/db/migration")
    if (migrationDir.exists()) migrationDir.deleteRecursively()

    val paths = listOf("config", "api/dto", "repository", "service", "api", "model")
    paths.forEach { File("$TARGET_DIR/io/koraframework/benchmark/java/$it").mkdirs() }
    migrationDir.mkdirs()

    // 1. ГЕНЕРАЦИЯ КОНФИГУРАЦИЙ
    if (includeConfigs && configsCount > 0) {
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
    }

    // 2. ГЕНЕРАЦИЯ DTO
    for (i in 0 until controllersCount) {
        val dtoFile = File("$TARGET_DIR/io/koraframework/benchmark/java/api/dto/OrderDto$i.java")
        dtoFile.writeText(
            """
            package io.koraframework.benchmark.java.api.dto;
            
            import io.koraframework.json.common.annotation.Json;
            import io.koraframework.validation.common.annotation.*;
            import java.math.BigDecimal;
            import java.util.List;
            
            @Valid
            public final class OrderDto$i {
                
                @Valid
                @Json 
                public record ItemRequest$i(
                    @NotEmpty
                    @Pattern("^[A-Z]{3,10}$")
                    String typeCode,
                    
                    @Min(1)
                    Integer quantity
                ) {}

                @Valid
                @Json 
                public record CreateOrderRequest$i(
                    Long userId,
                    
                    @NotEmpty
                    @Pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$")
                    String userEmail,
                    
                    @NotEmpty
                    @Size(min = 5, max = 15)
                    String promoCode,
                    
                    @NotEmpty
                    @Size(max = 50)
                    List<@Valid ItemRequest$i> items,
                    
                    @Size(min = 1, max = 5)
                    List<@NotEmpty String> tags
                ) {}
                
                @Valid
                @Json 
                public record OrderResponse$i(
                    @NotEmpty
                    @Size(min = 10, max = 50)
                    String id, 
                    
                    Long userId, 
                    
                    BigDecimal totalAmount, 
                    
                    @NotEmpty
                    String status
                ) {}
            }
            """.trimIndent()
        )
    }

    // 3. ГЕНЕРАЦИЯ СУЩНОСТЕЙ (MODEL) И РЕПОЗИТОРИЕВ
    val sqlBuilder = StringBuilder()
    if (includeRepos && reposCount > 0) {
        for (i in 0 until reposCount) {
            // 3.1 Генерируем Entity
            val entityFile = File("$TARGET_DIR/io/koraframework/benchmark/java/model/OrderEntity$i.java")
            entityFile.writeText(
                """
                package io.koraframework.benchmark.java.model;
                
                import io.koraframework.database.jdbc.annotation.*;
                import java.math.BigDecimal;
                
                @EntityJdbc
                public final class OrderEntity$i {
                    private String id;
                    private Long userId;
                    private BigDecimal totalAmount;
                    private String status;
                    
                    public OrderEntity$i() {}
                    
                    public OrderEntity$i(String id, Long userId, BigDecimal totalAmount, String status) {
                        this.id = id;
                        this.userId = userId;
                        this.totalAmount = totalAmount;
                        this.status = status;
                    }
                    
                    public String getId() { return id; }
                    public void setId(String id) { this.id = id; }
                    
                    public Long getUserId() { return userId; }
                    public void setUserId(Long userId) { this.userId = userId; }
                    
                    public BigDecimal getTotalAmount() { return totalAmount; }
                    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
                    
                    public String getStatus() { return status; }
                    public void setStatus(String status) { this.status = status; }
                }
                """.trimIndent()
            )

            // 3.2 Генерируем Repository, работающий с этой Entity
            val repoFile = File("$TARGET_DIR/io/koraframework/benchmark/java/repository/OrderJdbcRepository$i.java")
            repoFile.writeText(
                """
                package io.koraframework.benchmark.java.repository;
                
                import io.koraframework.benchmark.java.model.OrderEntity$i;
                import io.koraframework.database.common.annotation.Query;
                import io.koraframework.database.common.annotation.Repository;
                import io.koraframework.database.jdbc.JdbcRepository;
                
                @Repository
                public interface OrderJdbcRepository$i extends JdbcRepository {
                    
                    @Query("SELECT id, user_id, total_amount, status FROM orders_$i WHERE id = :id")
                    OrderEntity$i findById(String id);
                    
                    @Query("INSERT INTO orders_$i(id, user_id, total_amount, status) VALUES (:entity.id, :entity.userId, :entity.totalAmount, :entity.status)")
                    void save(OrderEntity$i entity);
                }
                """.trimIndent()
            )

            // 3.3 SQL миграция
            sqlBuilder.append(
                """
                CREATE TABLE orders_$i (
                    id VARCHAR(50) PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    total_amount NUMERIC(19, 4) NOT NULL,
                    status VARCHAR(50) NOT NULL
                );
                INSERT INTO orders_$i (id, user_id, total_amount, status) VALUES ('ORD-INIT-$i', 1, 15000.75, 'PAID');
                """.trimIndent()
            ).append("\n\n")
        }
        val migrationFile = File(migrationDir, "V20261018131100__create_benchmark_orders_tables.sql")
        migrationFile.writeText(sqlBuilder.toString())
    }

    // 4. ГЕНЕРАЦИЯ СЕРВИСОВ И МАППЕРОВ (СЛОЖНЫЙ СЛУЧАЙНЫЙ DAG)
    if (includeServices && servicesCount > 0) {
        val random = java.util.Random(42) // Фиксированный сид для воспроизводимости бенчмарков
        val maxServiceDeps = 5            // Максимум до 5 зависимостей между сервисами

        // 4.1 Генерируем пул переиспользуемых Мапперов
        if (includeMappers && mappersCount > 0) {
            for (i in 0 until mappersCount) {
                val mapperFile = File("$TARGET_DIR/io/koraframework/benchmark/java/service/OrderMapper$i.java")
                mapperFile.writeText(
                    """
                package io.koraframework.benchmark.java.service;
                
                import io.koraframework.common.annotation.Component;
                
                @Component
                public class OrderMapper$i {
                    // Принимает Object, чтобы работать с любой сгенерированной сущностью OrderEntityX
                    public String mapToDto(Object entity) {
                        if (entity == null) return null;
                        return "MAPPED_DTO_[" + entity.toString() + "]_BY_MAPPER_$i";
                    }
                
                    public String mapToEntity(String dto) {
                        if (dto == null) return null;
                        return "MAPPED_ENTITY_[" + dto + "]_BY_MAPPER_$i";
                    }
                }
                """.trimIndent()
                )
            }
        }

        // 4.2 Генерация Сервисов со случайными связями (до 5) и мапперами
        for (i in 0 until servicesCount) {
            val configId = if (includeConfigs && configsCount > 0) i % configsCount else -1

            val fields = mutableListOf<String>()
            val constructorParams = mutableListOf<String>()
            val constructorAssignments = mutableListOf<String>()
            val imports = mutableSetOf<String>()

            // Подключаем маппер из пула (детерминированно распределяем по сервисам)
            val mapperId = if (includeMappers && mappersCount > 0) i % mappersCount else -1
            if (mapperId != -1) {
                fields.add("private final OrderMapper$mapperId mapper;")
                constructorParams.add("OrderMapper$mapperId mapper")
                constructorAssignments.add("this.mapper = mapper;")
                // Если мапперы лежат в том же пакете, импорт не нужен, но на случай другой подпапки сохраняем структуру
                imports.add("import io.koraframework.benchmark.java.service.OrderMapper$mapperId;")
            }

            // Подключение конфигурации
            if (includeConfigs && configId != -1) {
                fields.add("private final SystemConfig$configId config;")
                constructorParams.add("SystemConfig$configId config")
                constructorAssignments.add("this.config = config;")
                imports.add("import io.koraframework.benchmark.java.config.SystemConfig$configId;")
            }

            // Подключение репозиториев
            val serviceReposIds = if (includeRepos && reposCount > 0) {
                val count = 1 + (i % 3) // 1-3 репозитория
                (0 until count).map { (i + it) % reposCount }.distinct()
            } else emptyList()

            serviceReposIds.forEach { id ->
                fields.add("private final OrderJdbcRepository$id repository$id;")
                constructorParams.add("OrderJdbcRepository$id repository$id")
                constructorAssignments.add("this.repository$id = repository$id;")
                imports.add("import io.koraframework.benchmark.java.repository.OrderJdbcRepository$id;")
            }

            // РЕАЛИСТИЧНЫЕ СВЯЗИ: Сервис i зависит от 1..5 случайных ПРЕДЫДУЩИХ сервисов (j < i)
            val childServicesCalls = mutableListOf<String>()
            val childServiceFieldNames = mutableListOf<String>()

            if (i > 0) {
                val depsCount = random.nextInt(minOf(i, maxServiceDeps)) + 1
                val chosenDeps = mutableSetOf<Int>()

                while (chosenDeps.size < depsCount && chosenDeps.size < i) {
                    chosenDeps.add(random.nextInt(i))
                }

                chosenDeps.sorted().forEach { depId ->
                    fields.add("private final OrderService$depId childService$depId;")
                    constructorParams.add("OrderService$depId childService$depId")
                    constructorAssignments.add("this.childService$depId = childService$depId;")
                    childServicesCalls.add("this.childService$depId.getOrder(id)")
                    childServiceFieldNames.add("childService$depId")
                    imports.add("import io.koraframework.benchmark.java.service.OrderService$depId;")
                }
            }

            // Логика методов с безопасной проверкой наличия маппера
            val synchronousChain = buildString {
                val mapperPrefix = if (mapperId != -1) "this.mapper.mapToDto(" else ""
                val mapperSuffix = if (mapperId != -1) ")" else ""

                if (includeRepos && serviceReposIds.isNotEmpty()) {
                    serviceReposIds.forEachIndexed { idx, id ->
                        appendLine("        io.koraframework.benchmark.java.model.OrderEntity$id repoRes$idx = this.repository$id.findById(id);")
                        appendLine("        if (repoRes$idx != null) return ${mapperPrefix}repoRes$idx${mapperSuffix};")
                    }
                }
                if (childServicesCalls.isNotEmpty()) {
                    childServicesCalls.forEachIndexed { idx, call ->
                        appendLine("        String childRes$idx = $call;")
                        appendLine("        if (childRes$idx != null) return ${mapperPrefix}childRes$idx${mapperSuffix};")
                    }
                }
                appendLine("        return ${mapperPrefix}\"DUMMY_ORDER_FROM_SERVICE_$i\"${mapperSuffix};")
            }

            // Сборка и запись файла Сервиса
            val serviceFile = File("$TARGET_DIR/io/koraframework/benchmark/java/service/OrderService$i.java")
            serviceReposIds.forEach { id ->
                imports.add("import io.koraframework.benchmark.java.model.OrderEntity$id;")
            }

            val importsStr = imports.sorted().joinToString("\n")
            val fieldsStr = fields.joinToString("\n    ")
            val paramsStr = constructorParams.joinToString(", ")
            val assignmentsStr = constructorAssignments.joinToString("\n        ")

            val firstRepoId = if (serviceReposIds.isNotEmpty()) serviceReposIds.first() else -1

            serviceFile.writeText(buildString {
                appendLine("package io.koraframework.benchmark.java.service;")
                appendLine()
                if (importsStr.isNotEmpty()) appendLine(importsStr)
                appendLine("import io.koraframework.common.annotation.Component;")
                appendLine("import java.util.concurrent.ThreadLocalRandom;")
                appendLine()
                appendLine("@Component")
                appendLine("public class OrderService$i {")
                if (fieldsStr.isNotEmpty()) appendLine("    $fieldsStr")
                appendLine()
                appendLine("    public OrderService$i($paramsStr) {")
                if (assignmentsStr.isNotEmpty()) appendLine("        $assignmentsStr")
                appendLine("    }")
                appendLine()
                appendLine("    public String getOrder(String id) {")
                append(synchronousChain)
                appendLine("    }")
                appendLine()
                appendLine("    public String createOrder(int quantity) {")
                appendLine("        String id = String.valueOf(ThreadLocalRandom.current().nextLong(1_000_000_000L));")

                if (mapperId != -1) {
                    appendLine("        String mappedId = this.mapper.mapToEntity(id);")
                    if (includeRepos && firstRepoId != -1) {
                        appendLine("        OrderEntity$firstRepoId entity = new OrderEntity$firstRepoId(mappedId, 1L, java.math.BigDecimal.valueOf(quantity), \"PENDING\");")
                        appendLine("        this.repository$firstRepoId.save(entity);")
                        appendLine("        return mappedId;")
                    } else if (childServiceFieldNames.isNotEmpty()) {
                        appendLine("        return this.${childServiceFieldNames.first()}.createOrder(quantity);")
                    } else {
                        appendLine("        return mappedId;")
                    }
                } else {
                    if (includeRepos && firstRepoId != -1) {
                        appendLine("        OrderEntity$firstRepoId entity = new OrderEntity$firstRepoId(id, 1L, java.math.BigDecimal.valueOf(quantity), \"PENDING\");")
                        appendLine("        this.repository$firstRepoId.save(entity);")
                        appendLine("        return id;")
                    } else if (childServiceFieldNames.isNotEmpty()) {
                        appendLine("        return this.${childServiceFieldNames.first()}.createOrder(quantity);")
                    } else {
                        appendLine("        return id;")
                    }
                }
                appendLine("    }")
                appendLine("}")
            })
        }
    }

    // 5. ГЕНЕРАЦИЯ КОНТРОЛЛЕРОВ С УЧЕТОМ НОВОЙ СТРУКТУРЫ DTO
    for (i in 0 until controllersCount) {
        val configId = if (includeConfigs && configsCount > 0) i % configsCount else -1
        val fields = mutableListOf<String>()
        val constructorParams = mutableListOf<String>()
        val constructorAssignments = mutableListOf<String>()
        val imports = mutableSetOf<String>()

        if (includeConfigs && configId != -1) {
            fields.add("private final SystemConfig$configId config;")
            constructorParams.add("SystemConfig$configId config")
            constructorAssignments.add("this.config = config;")
            imports.add("import io.koraframework.benchmark.java.config.SystemConfig$configId;")
        }

        // Контроллеры должны биндиться СТРОГО к СЛОЮ 0 сервисов (Точки входа в API)
        val controllerServicesIds = if (includeServices && servicesCount > 0) {
            val layersCount = 3
            val servicesPerLayer = (servicesCount + layersCount - 1) / layersCount
            val layer0Size = minOf(servicesCount, servicesPerLayer)

            val count = 1 + (i % 2) // Каждому контроллеру даем 1-2 фасада из Слоя 0
            (0 until count).map { (i + it) % layer0Size }.distinct()
        } else emptyList()

        controllerServicesIds.forEach { id ->
            fields.add("private final OrderService$id orderService$id;")
            constructorParams.add("OrderService$id orderService$id")
            constructorAssignments.add("this.orderService$id = orderService$id;")
            imports.add("import io.koraframework.benchmark.java.service.OrderService$id;")
        }

        val controllerChain = buildString {
            if (includeServices && controllerServicesIds.isNotEmpty()) {
                controllerServicesIds.forEachIndexed { idx, id ->
                    appendLine("        String sRes$idx = this.orderService$id.getOrder(id);")
                    appendLine("        if (sRes$idx != null) return new OrderResponse$i(sRes$idx, 42L, java.math.BigDecimal.TEN, \"GENERATED\");")
                }
                appendLine("        return null;")
            } else {
                appendLine("        return new OrderResponse$i(id, 42L, java.math.BigDecimal.TEN, \"GENERATED\");")
            }
        }

        val file = File("$TARGET_DIR/io/koraframework/benchmark/java/api/OrderController$i.java")
        file.writeText(buildString {
            appendLine("package io.koraframework.benchmark.java.api;")
            appendLine()
            appendLine("import io.koraframework.benchmark.java.api.dto.OrderDto$i.*;")
            imports.sorted().forEach { appendLine(it) }
            appendLine("import io.koraframework.common.annotation.Component;")
            appendLine("import io.koraframework.http.common.HttpMethod;")
            appendLine("import io.koraframework.http.common.annotation.HttpRoute;")
            appendLine("import io.koraframework.http.common.annotation.Path;")
            appendLine("import io.koraframework.http.server.common.annotation.HttpController;")
            appendLine("import io.koraframework.json.common.annotation.Json;")
            appendLine("import java.math.BigDecimal;")
            appendLine()
            appendLine("@HttpController")
            appendLine("@Component")
            appendLine("public class OrderController$i {")
            appendLine()
            if (fields.isNotEmpty()) fields.forEach { appendLine("    $it") }
            appendLine()
            appendLine("    public OrderController$i(${constructorParams.joinToString(", ")}) {")
            if (constructorAssignments.isNotEmpty()) constructorAssignments.forEach { appendLine("        $it") }
            appendLine("    }")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.GET, path = \"/orders/$i/{id}\")")
            appendLine("    public OrderResponse$i getOrder(@Path String id) {")
            append(controllerChain)
            appendLine("    }")
            appendLine()
            appendLine("    @Json")
            appendLine("    @HttpRoute(method = HttpMethod.POST, path = \"/orders/$i\")")
            appendLine("    public OrderResponse$i checkout(@Json CreateOrderRequest$i body) {")
            appendLine("        int totalQuantity = 0;")
            appendLine("        if (body.items() != null) {")
            appendLine("            totalQuantity = body.items().stream()")
            appendLine("                .filter(item -> item != null && item.quantity() != null)")
            appendLine("                .mapToInt(ItemRequest$i::quantity)")
            appendLine("                .sum();")
            appendLine("        }")
            appendLine()

            if (includeServices && controllerServicesIds.isNotEmpty()) {
                appendLine("        String entity = this.orderService${controllerServicesIds.first()}.createOrder(totalQuantity);")
                appendLine("        return new OrderResponse$i(entity, body.userId(), java.math.BigDecimal.TEN, \"GENERATED\");")
            } else {
                appendLine("        return new OrderResponse$i(\"GENERATED_DIRECTLY_\" + totalQuantity, body.userId(), java.math.BigDecimal.TEN, \"GENERATED\");")
            }
            appendLine("    }")
            appendLine("}")
        })
    }

    val appFile = File("$TARGET_DIR/io/koraframework/benchmark/java/App.java")
    val hasRepos = includeRepos && reposCount > 0
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
        import io.koraframework.validation.module.ValidationModule;
        ${if (hasRepos) "import io.koraframework.database.jdbc.JdbcDatabaseModule;" else ""}
        ${if (hasRepos) "import io.koraframework.database.flyway.FlywayJdbcDatabaseModule;" else ""}
        
        @KoraApp
        public interface App extends
            HoconConfigModule,
            JsonModule,
            LoggingModule,
            LogbackModule,
            ValidationModule,
            UndertowPublicHttpServerModule${if (hasRepos) ",\n            JdbcDatabaseModule,\n            FlywayJdbcDatabaseModule" else ""} {
            static void main(String[] args) {
                KoraApplication.run(AppGraph::graph);
            }
        }
        """.trimIndent()
    )

    // 7. ГЕНЕРАЦИЯ APPLICATION.CONF
    val configContent = buildString {
        appendLine("app { name = \"Kora Benchmark\" \n version = \"1\" \n environment = \"dev\" }")
        if (includeConfigs && configsCount > 0) {
            appendLine("system {")
            for (i in 0 until configsCount) {
                appendLine("  module$i { environment = \"prod\" \n version = \"2\" }")
            }
            appendLine("}")
        }
        appendLine(
            """
        httpServer {
          port = 8090
          undertow {
            ioThreads = 4
          }
          headerServerDateEnabled = false
          telemetry.metrics.enabled = false
          telemetry.logging.enabled = false
          telemetry.tracing.enabled = false
        }
        """.trimIndent()
        )
        if (hasRepos) {
            appendLine(
                """
            jdbc {
              jdbcUrl = "jdbc:postgresql://localhost:5432/service_kora"
              username = "service_kora"
              password = "1"
              poolName = "kora-pool"
            }
            flyway.enabled = true
            """.trimIndent()
            )
        }
    }

    File(RESOURCES_DIR, "application.conf").writeText(configContent)
    println("Генерация завершена. Сгенерировано DI-нод: $totalNodes (Контроллеров: $controllersCount, Сервисов: $servicesCount, Репозиториев: $reposCount)")
}
