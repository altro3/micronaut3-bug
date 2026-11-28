package com.altro.servicekora

import java.io.*

class KoraGeneratorJava

fun main() {
    val TARGET_DIR = "./service-kora-benchmark-java/src/main/java"
    val RESOURCES_DIR = "./service-kora-benchmark-java/src/main/resources"

    val includeConfigs = true
    val includeRepos = false
    val includeMappers = true
    val includeServices = true

    val configsPart = 0.10
    val reposPart = 0.13
    val mappersPart = 0.12
    val controllersPart = 0.10

    val totalNodes = 150

    val configsCount = if (includeConfigs) maxOf(1, (totalNodes * configsPart).toInt()) else 0
    val controllersCount = if (includeServices) maxOf(1, (totalNodes * controllersPart).toInt()) else 0
    val reposCount = if (includeRepos) maxOf(1, (totalNodes * reposPart).toInt()) else 0
    val mappersCount = if (includeMappers) maxOf(1, (totalNodes * mappersPart).toInt()) else 0
    val servicesCount = maxOf(1, totalNodes - configsCount - reposCount - controllersCount - mappersCount)

    val basePackageDir = File("$TARGET_DIR/io/koraframework/benchmark/java")
    if (basePackageDir.exists()) basePackageDir.deleteRecursively()

    val migrationDir = File("$RESOURCES_DIR/db/migration")
    if (migrationDir.exists()) migrationDir.deleteRecursively()

    val paths = listOf("config", "api/dto", "repository", "service", "api", "model", "bench", "bench/dto")
    paths.forEach { File("$TARGET_DIR/io/koraframework/benchmark/java/$it").mkdirs() }
    if (includeRepos && reposCount > 0) {
        migrationDir.mkdirs()
    }

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

    // 2. ГЕНЕРАЦИЯ DTO ОСНОВНОГО ГРАФА
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
    // 3. ГЕНЕРАЦИЯ СУЩНОСТЕЙ И РЕПОЗИТОРИЕВ (ТОЛЬКО ПРИ includeRepos = true)
    val sqlBuilder = StringBuilder()
    if (includeRepos && reposCount > 0) {
        for (i in 0 until reposCount) {
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
        val random = java.util.Random(42)
        val maxServiceDeps = 5

        if (includeMappers && mappersCount > 0) {
            for (i in 0 until mappersCount) {
                val mapperFile = File("$TARGET_DIR/io/koraframework/benchmark/java/service/OrderMapper$i.java")
                mapperFile.writeText(
                    """
                    package io.koraframework.benchmark.java.service;

                    import io.koraframework.common.annotation.Component;

                    @Component
                    public class OrderMapper$i {
                        public String mapToDto(Object entity) {
                            if (entity == null) return null;
                            return "MAPPED_DTO_[" + entity + "]_BY_MAPPER_$i";
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
        for (i in 0 until servicesCount) {
            val configId = if (includeConfigs && configsCount > 0) i % configsCount else -1

            val fields = mutableListOf<String>()
            val constructorParams = mutableListOf<String>()
            val constructorAssignments = mutableListOf<String>()
            val imports = mutableSetOf<String>()

            val mapperId = if (includeMappers && mappersCount > 0) i % mappersCount else -1
            if (mapperId != -1) {
                fields.add("private final OrderMapper$mapperId mapper;")
                constructorParams.add("OrderMapper$mapperId mapper")
                constructorAssignments.add("this.mapper = mapper;")
                imports.add("import io.koraframework.benchmark.java.service.OrderMapper$mapperId;")
            }

            if (includeConfigs && configId != -1) {
                fields.add("private final SystemConfig$configId config;")
                constructorParams.add("SystemConfig$configId config")
                constructorAssignments.add("this.config = config;")
                imports.add("import io.koraframework.benchmark.java.config.SystemConfig$configId;")
            }

            val serviceReposIds = if (includeRepos && reposCount > 0) {
                val count = 1 + (i % 3)
                (0 until count).map { (i + it) % reposCount }.distinct()
            } else emptyList()

            serviceReposIds.forEach { id ->
                fields.add("private final OrderJdbcRepository$id repository$id;")
                constructorParams.add("OrderJdbcRepository$id repository$id")
                constructorAssignments.add("this.repository$id = repository$id;")
                imports.add("import io.koraframework.benchmark.java.repository.OrderJdbcRepository$id;")
            }

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
                    childServicesCalls.add("childService$depId.getOrder(id)")
                    childServiceFieldNames.add("childService$depId")
                    imports.add("import io.koraframework.benchmark.java.service.OrderService$depId;")
                }
            }

            val synchronousChain = buildString {
                val mapperPrefix = if (mapperId != -1) "mapper.mapToDto(" else ""
                val mapperSuffix = if (mapperId != -1) ")" else ""

                if (includeRepos && serviceReposIds.isNotEmpty()) {
                    serviceReposIds.forEachIndexed { idx, id ->
                        appendLine("        var repoRes$idx = repository$id.findById(id);")
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
                appendLine("import java.util.concurrent.atomic.AtomicLong;")
                appendLine()
                appendLine("@Component")
                appendLine("public class OrderService$i {")
                appendLine("    private static final AtomicLong ID_SEQ = new AtomicLong(ThreadLocalRandom.current().nextLong(1_000_000_000_000L));")
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
                appendLine("        String id = String.valueOf(ID_SEQ.incrementAndGet());")

                if (mapperId != -1) {
                    appendLine("        String mappedId = mapper.mapToEntity(id);")
                    if (includeRepos && firstRepoId != -1) {
                        appendLine("        var entity = new OrderEntity$firstRepoId(mappedId, 1L, java.math.BigDecimal.valueOf(quantity), \"PENDING\");")
                        appendLine("        repository$firstRepoId.save(entity);")
                        appendLine("        return mappedId;")
                    } else if (childServiceFieldNames.isNotEmpty()) {
                        appendLine("        return ${childServiceFieldNames.first()}.createOrder(quantity);")
                    } else {
                        appendLine("        return mappedId;")
                    }
                } else {
                    if (includeRepos && firstRepoId != -1) {
                        appendLine("        var entity = new OrderEntity$firstRepoId(id, 1L, java.math.BigDecimal.valueOf(quantity), \"PENDING\");")
                        appendLine("        repository$firstRepoId.save(entity);")
                        appendLine("        return id;")
                    } else if (childServiceFieldNames.isNotEmpty()) {
                        appendLine("        return ${childServiceFieldNames.first()}.createOrder(quantity);")
                    } else {
                        appendLine("        return id;")
                    }
                }
                appendLine("    }")
                appendLine("}")
            })
        }
    }
    // 5. ГЕНЕРАЦИЯ КОНТРОЛЛЕРОВ ОСНОВНОГО ГРАФА
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

        val controllerServicesIds = if (includeServices && servicesCount > 0) {
            val layersCount = 3
            val servicesPerLayer = (servicesCount + layersCount - 1) / layersCount
            val layer0Size = minOf(servicesCount, servicesPerLayer)

            val count = 1 + (i % 2)
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
                    appendLine("        String sRes$idx = orderService$id.getOrder(id);")
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
                appendLine("        String entity = orderService${controllerServicesIds.first()}.createOrder(totalQuantity);")
                appendLine("        return new OrderResponse$i(entity, body.userId(), java.math.BigDecimal.TEN, \"GENERATED\");")
            } else {
                appendLine("        return new OrderResponse$i(\"GENERATED_DIRECTLY_\" + totalQuantity, body.userId(), java.math.BigDecimal.TEN, \"GENERATED\");")
            }
            appendLine("    }")
            appendLine("}")
        })
    }

    // 6. ГЕНЕРАЦИЯ 3 ОБОСОБЛЕННЫХ ЭТАЛОННЫХ БЕНЧМАРК-ЭНДПОИНТОВ (ВСЕГДА ПРИСУТСТВУЮТ, БЕЗ БД)
    generateIsolatedBenchmarks(TARGET_DIR)

    // 7. ГЕНЕРАЦИЯ APP.JAVA
    val appFile = File("$TARGET_DIR/io/koraframework/benchmark/java/App.java")
    val hasRepos = includeRepos && reposCount > 0
    appFile.writeText(
        """
        package io.koraframework.benchmark.java;

        import io.koraframework.application.graph.*;
        import io.koraframework.common.annotation.*;
        import io.koraframework.config.hocon.*;
        import io.koraframework.http.server.undertow.*;
        import io.koraframework.json.common.*;
        import io.koraframework.logging.common.*;
        import io.koraframework.logging.logback.*;
        import io.koraframework.validation.module.*;
        //import io.koraframework.config.aot.generated.AotModule;
        //import io.koraframework.http.server.netty.NettyPublicHttpServerModule;
        ${if (hasRepos) "import io.koraframework.database.jdbc.JdbcDatabaseModule;" else ""}
        ${if (hasRepos) "import io.koraframework.database.flyway.FlywayJdbcDatabaseModule;" else ""}

        @KoraApp
        public interface App extends
        //    AotModule,
        //    NettyPublicHttpServerModule,
            HoconConfigModule,
            UndertowPublicHttpServerModule,${if (hasRepos) "\n            FlywayJdbcDatabaseModule," else ""}
            JsonModule,
            LoggingModule,
            LogbackModule,
            ValidationModule${if (hasRepos) ",\n            JdbcDatabaseModule" else ""} {
            static void main(String[] args) {
                KoraApplication.run(AppGraph::graph);
            }
        }
        """.trimIndent()
    )

    // 8. ГЕНЕРАЦИЯ APPLICATION.CONF
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
                  maxPoolSize = 64
                  minIdle = 32
                }
                flyway.enabled = true
                """.trimIndent()
            )
        }
    }

    File(RESOURCES_DIR, "application.conf").writeText(configContent)
    println("Генерация завершена. Сгенерировано DI-нод: $totalNodes (Контроллеров: $controllersCount, Сервисов: $servicesCount, Репозиториев: $reposCount) + 3 эталонных бенчмарк-эндпоинта (/bench/small, /bench/medium, /bench/large)")
}

private fun generateIsolatedBenchmarks(targetDir: String) {
    // 1. DTO ДЛЯ SMALL (~100 байт), MEDIUM (~1.5-2 КБ), LARGE (~15-25 КБ)
    // 1. DTO ДЛЯ SMALL (~100 байт), MEDIUM (~1.5-2 КБ), LARGE (~15-25 КБ), EXTRABIG (~70 КБ)
    val benchDtoFile = File("$targetDir/io/koraframework/benchmark/java/bench/dto/BenchmarkDtos.java")
    benchDtoFile.writeText(
        """
        package io.koraframework.benchmark.java.bench.dto;

        import io.koraframework.json.common.annotation.Json;
        import io.koraframework.json.common.annotation.JsonReader;
        import io.koraframework.json.common.annotation.JsonWriter;
        import io.koraframework.validation.common.annotation.*;
        import java.math.BigDecimal;
        import java.math.BigInteger;
        import java.time.*;
        import java.util.List;
        import java.util.Map;
        import java.util.Set;
        import java.util.UUID;

        @Valid
        public final class BenchmarkDtos {

            // ==================== 1. SMALL (~100 bytes) ====================
            @Valid @JsonReader
            public record SmallItemRequest(
                @NotEmpty @Pattern("^[A-Z]{3,10}$") String typeCode,
                @Min(1) Integer quantity
            ) {}

            @Valid @JsonReader
            public record SmallOrderRequest(
                Long userId,
                @NotEmpty @Pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$") String userEmail,
                @NotEmpty @Size(min = 5, max = 15) String promoCode,
                @NotEmpty @Size(max = 50) List<@Valid SmallItemRequest> items,
                @Size(min = 1, max = 5) List<@NotEmpty String> tags
            ) {}

            @Valid @JsonWriter
            public record SmallOrderResponse(
                @NotEmpty @Size(min = 10, max = 50) String id,
                Long userId,
                BigDecimal totalAmount,
                @NotEmpty String status
            ) {}

            // ==================== 2. MEDIUM (~1.5 - 2 KB) ====================
            @Valid @Json
            public record CustomerProfile(
                @NotEmpty String fullName,
                @NotEmpty @Pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$") String email,
                @NotEmpty String phone,
                @NotEmpty String loyaltyTier,
                int loyaltyPoints,
                Map<String, String> preferences
            ) {}

            @Valid @Json
            public record DeliveryAddress(
                @NotEmpty String countryCode,
                @NotEmpty String city,
                @NotEmpty String street,
                @NotEmpty String building,
                String apartment,
                @NotEmpty @Size(min = 4, max = 12) String postalCode,
                double latitude,
                double longitude,
                String courierComment
            ) {}

            @Valid @Json
            public record MediumOrderItem(
                @NotEmpty @Pattern("^[A-Z0-9-]{4,20}$") String sku,
                @NotEmpty String title,
                @NotEmpty String category,
                @Min(1) int quantity,
                BigDecimal unitPrice,
                BigDecimal discountAmount,
                boolean giftWrap,
                List<String> appliedPromos
            ) {}

            @Valid @JsonReader
            public record MediumOrderRequest(
                Long userId,
                @NotEmpty String clientRequestId,
                @NotEmpty String currency,
                @NotEmpty String paymentMethod,
                @Size(min = 3, max = 20) String promoCode,
                @Valid CustomerProfile customer,
                @Valid DeliveryAddress deliveryAddress,
                @NotEmpty @Size(min = 1, max = 50) List<@Valid MediumOrderItem> items,
                List<String> tags,
                Map<String, String> metadata
            ) {}

            @Valid @JsonWriter
            public record MediumOrderResponse(
                @NotEmpty String orderId,
                Long userId,
                @NotEmpty String status,
                @NotEmpty String currency,
                BigDecimal subtotalAmount,
                BigDecimal discountTotal,
                BigDecimal deliveryFee,
                BigDecimal finalAmount,
                int totalItemsCount,
                @Valid CustomerProfile customer,
                @Valid DeliveryAddress deliveryAddress,
                List<@Valid MediumOrderItem> processedItems,
                List<String> auditTrail
            ) {}

            // ==================== 3. LARGE (~15 - 25 KB) ====================
            @Valid @Json
            public record ItemAttribute(
                @NotEmpty String key,
                @NotEmpty String value,
                String unit,
                boolean searchable
            ) {}

            @Valid @Json
            public record WarehouseAllocation(
                @NotEmpty String warehouseCode,
                @NotEmpty String region,
                @Min(1) int reservedQty,
                int estimatedDeliveryHours
            ) {}

            @Valid @Json
            public record LargeOrderItem(
                @NotEmpty @Pattern("^[A-Z0-9-]{4,24}$") String sku,
                @NotEmpty String productName,
                @NotEmpty String manufacturer,
                @NotEmpty String categoryPath,
                @NotEmpty String description,
                @Min(1) int quantity,
                BigDecimal basePrice,
                BigDecimal discountAmount,
                BigDecimal taxRate,
                BigDecimal taxAmount,
                BigDecimal lineTotal,
                double weightKg,
                List<@Valid ItemAttribute> attributes,
                List<@Valid WarehouseAllocation> allocations,
                List<String> tags
            ) {}

            @Valid @Json
            public record AuditEvent(
                @NotEmpty String eventId,
                @NotEmpty String actor,
                @NotEmpty String action,
                @NotEmpty String timestampIso,
                String details
            ) {}

            @Valid @JsonReader
            public record LargeOrderRequest(
                Long userId,
                @NotEmpty String correlationId,
                @NotEmpty String tenantId,
                @NotEmpty String channel,
                @NotEmpty String currency,
                @NotEmpty String paymentMethod,
                @Valid CustomerProfile customer,
                @Valid DeliveryAddress billingAddress,
                @Valid DeliveryAddress shippingAddress,
                @NotEmpty @Size(min = 1, max = 200) List<@Valid LargeOrderItem> items,
                List<@Valid AuditEvent> history,
                Map<String, String> customHeaders,
                Map<String, String> analyticsContext
            ) {}

            @Valid @JsonWriter
            public record LargeOrderResponse(
                @NotEmpty String orderId,
                @NotEmpty String correlationId,
                Long userId,
                @NotEmpty String status,
                @NotEmpty String currency,
                BigDecimal grossAmount,
                BigDecimal totalDiscount,
                BigDecimal totalTax,
                BigDecimal shippingCost,
                BigDecimal netPayableAmount,
                double totalWeightKg,
                int totalUnits,
                @Valid CustomerProfile customer,
                @Valid DeliveryAddress shippingAddress,
                List<@Valid LargeOrderItem> items,
                List<@Valid AuditEvent> auditEvents,
                Map<String, String> summaryMetrics
            ) {}

            // ==================== 4. EXTRABIG (~70 KB, 10 shards x ~7 KB) ====================
            @Valid @Json
            public record GeoCoordinateNode(
                double lat,
                double lon,
                Float altitudeMeters,
                int accuracyCm,
                Instant capturedAt,
                String zoneCode
            ) {}

            @Valid @Json
            public record TelemetryPacket(
                UUID packetId,
                long sequenceNumber,
                short sensorGroupId,
                int protocolVersion,
                String priorityFlag,
                boolean encrypted,
                Boolean verified,
                List<Integer> signatureBytes,
                String endpointUri,
                Duration processingDuration,
                Duration retentionPeriod,
                LocalDate accountingDate,
                LocalTime windowStartTime,
                LocalDateTime localTimestamp,
                OffsetDateTime offsetTimestamp,
                ZonedDateTime zonedTimestamp,
                BigInteger rawCounter,
                BigDecimal calibratedValue,
                Set<String> capabilityFlags,
                Map<String, Double> sensorReadings,
                @Valid GeoCoordinateNode location
            ) {}

            @Valid @Json
            public record NestedComplianceBlock(
                @NotEmpty String regulationCode,
                @NotEmpty String description,
                List<@Valid TelemetryPacket> packets,
                Map<String, List<String>> ruleMatrix
            ) {}

            @Valid @Json
            public record ExtraBigShardPayload(
                @NotEmpty String shardId,
                UUID traceUuid,
                long revisionLong,
                Long optionalRevision,
                int partitionInt,
                Integer optionalPartition,
                short regionShort,
                Short optionalRegion,
                int flagsByte,
                Integer optionalFlags,
                double ratioDouble,
                Double optionalRatio,
                float scoreFloat,
                Float optionalScore,
                boolean activeBool,
                Boolean optionalActive,
                String gradeChar,
                String optionalGrade,
                BigDecimal shardAmount,
                BigInteger shardTotalHashes,
                Instant createdAt,
                LocalDate businessDate,
                LocalTime cutoffTime,
                LocalDateTime systemDateTime,
                OffsetDateTime partnerOffsetTime,
                ZonedDateTime auditZonedTime,
                Duration ttlDuration,
                Duration gracePeriod,
                String callbackUri,
                List<Integer> binaryPayload,
                Set<Integer> allowedBuckets,
                List<String> routingKeys,
                Map<String, String> attributes,
                @Valid CustomerProfile shardOwner,
                @Valid DeliveryAddress shardDestination,
                List<@Valid LargeOrderItem> shardItems,
                @Valid NestedComplianceBlock compliance
                ) {}

                @Valid @JsonReader
                public record ExtraBigOrderRequest(
                    Long userId,
                    @NotEmpty String batchId,
                    @NotEmpty String currency,
                    Instant submittedAt,
                    @Valid ExtraBigShardPayload shard01,
                    @Valid ExtraBigShardPayload shard02,
                    @Valid ExtraBigShardPayload shard03,
                    @Valid ExtraBigShardPayload shard04,
                    @Valid ExtraBigShardPayload shard05,
                    @Valid ExtraBigShardPayload shard06,
                    @Valid ExtraBigShardPayload shard07,
                    @Valid ExtraBigShardPayload shard08,
                    @Valid ExtraBigShardPayload shard09,
                    @Valid ExtraBigShardPayload shard10
                ) {}

                @Valid @JsonWriter
                public record ExtraBigOrderResponse(
                    @NotEmpty String batchOrderId,
                    @NotEmpty String batchId,
                    Long userId,
                    @NotEmpty String status,
                    Instant processedAt,
                    BigDecimal grandTotalAmount,
                    BigInteger grandTotalHashes,
                    int totalShardsProcessed,
                    @Valid ExtraBigShardPayload shard01,
                    @Valid ExtraBigShardPayload shard02,
                    @Valid ExtraBigShardPayload shard03,
                    @Valid ExtraBigShardPayload shard04,
                    @Valid ExtraBigShardPayload shard05,
                    @Valid ExtraBigShardPayload shard06,
                    @Valid ExtraBigShardPayload shard07,
                    @Valid ExtraBigShardPayload shard08,
                    @Valid ExtraBigShardPayload shard09,
                    @Valid ExtraBigShardPayload shard10
                ) {}
        }
        """.trimIndent()
    )
    // 2. ИЗОЛИРОВАННЫЙ СЕРВИС С ПСЕВДОЛОГИКОЙ В ПАМЯТИ (БЕЗ БД)
    val benchServiceFile = File("$targetDir/io/koraframework/benchmark/java/bench/IsolatedBenchmarkService.java")
    benchServiceFile.writeText(
        """
        package io.koraframework.benchmark.java.bench;

        import io.koraframework.benchmark.java.bench.dto.BenchmarkDtos.*;
        import io.koraframework.common.annotation.Component;
        import java.math.BigDecimal;
        import java.math.RoundingMode;
        import java.util.ArrayList;
        import java.util.List;
        import java.util.Map;
        import java.util.concurrent.ThreadLocalRandom;
        import java.util.concurrent.atomic.AtomicLong;
        import java.math.BigInteger;
        import java.nio.charset.StandardCharsets;
        import java.net.URI;
        import java.net.URL;
        import java.time.*;
        import java.util.Set;
        import java.util.UUID;

        @Component
        public class IsolatedBenchmarkService {

            private static final AtomicLong ID_SEQ = new AtomicLong(ThreadLocalRandom.current().nextLong(1_000_000_000_000L));

            // --- SMALL ---
            public SmallOrderResponse getSmallOrder(String id) {
                return new SmallOrderResponse("ORD-SMALL-" + id, 42L, BigDecimal.TEN, "GENERATED");
            }

            public SmallOrderResponse processSmallOrder(SmallOrderRequest req) {
                int totalQty = 0;
                if (req.items() != null) {
                    for (var item : req.items()) {
                        if (item != null && item.quantity() != null) {
                            totalQty += item.quantity();
                        }
                    }
                }
                long seq = ID_SEQ.incrementAndGet();
                BigDecimal total = BigDecimal.valueOf(totalQty * 100L);
                return new SmallOrderResponse("ORD-S-" + seq, req.userId(), total, "CONFIRMED");
            }

            // --- MEDIUM ---
            public MediumOrderResponse getMediumOrder(String id) {
                var profile = new CustomerProfile(
                    "Ivan Petrov", "ivan.petrov@example.com", "+79991234567",
                    "GOLD", 1450, Map.of("lang", "ru", "theme", "dark", "notifications", "push")
                );
                var address = new DeliveryAddress(
                    "RU", "Novosibirsk", "Lenina", "12", "45",
                    "630099", 55.0302, 82.9204, "Call 15 minutes before arrival"
                );
                var items = List.of(
                    new MediumOrderItem("SKU-1001", "Mechanical Keyboard Pro", "ELECTRONICS", 1, new BigDecimal("12500.00"), new BigDecimal("1500.00"), true, List.of("WELCOME10")),
                    new MediumOrderItem("SKU-2045", "USB-C Braided Cable 2m", "ACCESSORIES", 2, new BigDecimal("1200.00"), BigDecimal.ZERO, false, List.of()),
                    new MediumOrderItem("SKU-3099", "Ergonomic Mouse Pad XL", "ACCESSORIES", 1, new BigDecimal("2400.00"), new BigDecimal("400.00"), false, List.of("BUNDLE5"))
                );
                return new MediumOrderResponse(
                    "ORD-MED-" + id, 10042L, "CONFIRMED", "RUB",
                    new BigDecimal("17300.00"), new BigDecimal("1900.00"), new BigDecimal("350.00"), new BigDecimal("15750.00"),
                    4, profile, address, items,
                    List.of("CREATED", "VALIDATED", "PRICED", "CONFIRMED")
                );
            }

            public MediumOrderResponse processMediumOrder(MediumOrderRequest req) {
                BigDecimal subtotal = BigDecimal.ZERO;
                BigDecimal discountTotal = BigDecimal.ZERO;
                int totalItems = 0;

                if (req.items() != null) {
                    for (var item : req.items()) {
                        BigDecimal qty = BigDecimal.valueOf(item.quantity());
                        subtotal = subtotal.add(item.unitPrice().multiply(qty));
                        if (item.discountAmount() != null) {
                            discountTotal = discountTotal.add(item.discountAmount().multiply(qty));
                        }
                        totalItems += item.quantity();
                    }
                }

                BigDecimal deliveryFee = subtotal.compareTo(new BigDecimal("5000")) >= 0 ? BigDecimal.ZERO : new BigDecimal("390.00");
                BigDecimal finalAmount = subtotal.subtract(discountTotal).add(deliveryFee);
                long seq = ID_SEQ.incrementAndGet();

                return new MediumOrderResponse(
                    "ORD-M-" + seq,
                    req.userId(),
                    "CONFIRMED",
                    req.currency(),
                    subtotal,
                    discountTotal,
                    deliveryFee,
                    finalAmount,
                    totalItems,
                    req.customer(),
                    req.deliveryAddress(),
                    req.items(),
                    List.of("RECEIVED:" + req.clientRequestId(), "FRAUD_CHECK_OK", "RESERVED", "CONFIRMED")
                );
            }
            // --- LARGE ---
            public LargeOrderResponse getLargeOrder(String id) {
                var items = new ArrayList<LargeOrderItem>(25);
                for (int i = 0; i < 25; i++) {
                    items.add(new LargeOrderItem(
                        "SKU-LRG-" + (1000 + i),
                        "Enterprise Server Component Module Revision " + i,
                        "HardwareCorp International",
                        "Infrastructure/Servers/Components/Tier" + (i % 4),
                        "High-reliability enterprise grade hardware component designed for continuous 24/7 data center workloads with extended thermal tolerance and telemetry diagnostics.",
                        2 + (i % 5),
                        new BigDecimal("45000.00"),
                        new BigDecimal("2500.00"),
                        new BigDecimal("0.20"),
                        new BigDecimal("8500.00"),
                        new BigDecimal("93500.00"),
                        3.45,
                        List.of(
                            new ItemAttribute("voltage", "220", "V", true),
                            new ItemAttribute("warranty", "36", "months", true),
                            new ItemAttribute("certification", "ISO-9001-REV2026", null, false)
                        ),
                        List.of(
                            new WarehouseAllocation("WH-NSK-01", "SIBERIA", 2, 12),
                            new WarehouseAllocation("WH-MSK-02", "CENTRAL", 3, 48)
                        ),
                        List.of("ENTERPRISE", "B2B", "FAST_SHIP", "INSURED")
                    ));
                }

                var profile = new CustomerProfile(
                    "Alexander Smirnov", "a.smirnov@enterprise-corp.ru", "+79130001122",
                    "PLATINUM_B2B", 98500, Map.of("contract", "B2B-2026-889", "sla", "4H", "accountManager", "E.Volkova")
                );
                var address = new DeliveryAddress(
                    "RU", "Novosibirsk", "Akademika Lavrentyeva", "6", "Office 401",
                    "630090", 54.8471, 83.1056, "Pass required at security gate B"
                );

                return new LargeOrderResponse(
                    "ORD-LRG-" + id, "CORR-" + id, 50001L, "APPROVED", "RUB",
                    new BigDecimal("2250000.00"), new BigDecimal("125000.00"), new BigDecimal("425000.00"),
                    BigDecimal.ZERO, new BigDecimal("2550000.00"), 86.25, 100,
                    profile, address, items,
                    List.of(
                        new AuditEvent("EVT-1", "SYSTEM", "CREATED", "2026-11-21T07:00:00Z", "Initial B2B payload ingested"),
                        new AuditEvent("EVT-2", "RISK_ENGINE", "APPROVED", "2026-11-21T07:00:01Z", "Credit limit verified")
                    ),
                    Map.of("processingNode", "kora-node-01", "itemsProcessed", "25", "slaTier", "PLATINUM")
                );
            }

            public LargeOrderResponse processLargeOrder(LargeOrderRequest req) {
                BigDecimal gross = BigDecimal.ZERO;
                BigDecimal totalDiscount = BigDecimal.ZERO;
                BigDecimal totalTax = BigDecimal.ZERO;
                double totalWeight = 0.0;
                int totalUnits = 0;

                if (req.items() != null) {
                    for (var item : req.items()) {
                        BigDecimal qty = BigDecimal.valueOf(item.quantity());
                        BigDecimal lineGross = item.basePrice().multiply(qty);
                        BigDecimal lineDisc = item.discountAmount() != null ? item.discountAmount().multiply(qty) : BigDecimal.ZERO;
                        BigDecimal taxable = lineGross.subtract(lineDisc);
                        BigDecimal lineTax = taxable.multiply(item.taxRate()).setScale(2, RoundingMode.HALF_UP);

                        gross = gross.add(lineGross);
                        totalDiscount = totalDiscount.add(lineDisc);
                        totalTax = totalTax.add(lineTax);
                        totalWeight += item.weightKg() * item.quantity();
                        totalUnits += item.quantity();
                    }
                }

                BigDecimal shippingCost = totalWeight > 50.0 ? new BigDecimal("2500.00") : BigDecimal.ZERO;
                BigDecimal netPayable = gross.subtract(totalDiscount).add(totalTax).add(shippingCost);
                long seq = ID_SEQ.incrementAndGet();

                return new LargeOrderResponse(
                    "ORD-L-" + seq,
                    req.correlationId(),
                    req.userId(),
                    "APPROVED",
                    req.currency(),
                    gross,
                    totalDiscount,
                    totalTax,
                    shippingCost,
                    netPayable,
                    totalWeight,
                    totalUnits,
                    req.customer(),
                    req.shippingAddress(),
                    req.items(),
                    req.history(),
                    Map.of("tenant", req.tenantId(), "channel", req.channel(), "units", String.valueOf(totalUnits))
                );
            }
            
            
            // --- EXTRABIG (~70 KB) ---
            public ExtraBigOrderResponse getExtraBigOrder(String id) {
                Instant now = Instant.parse("2026-11-28T06:00:00Z");
                var s01 = buildSampleShard(1, now);
                var s02 = buildSampleShard(2, now);
                var s03 = buildSampleShard(3, now);
                var s04 = buildSampleShard(4, now);
                var s05 = buildSampleShard(5, now);
                var s06 = buildSampleShard(6, now);
                var s07 = buildSampleShard(7, now);
                var s08 = buildSampleShard(8, now);
                var s09 = buildSampleShard(9, now);
                var s10 = buildSampleShard(10, now);

                return new ExtraBigOrderResponse(
                    "ORD-XBIG-" + id,
                    "BATCH-" + id,
                    90001L,
                    "COMPLETED",
                    now,
                    new BigDecimal("1548900.00"),
                    new BigInteger("987654321098765432109876543210"),
                    10,
                    s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
                );
            }

            public ExtraBigOrderResponse processExtraBigOrder(ExtraBigOrderRequest req) {
                BigDecimal grandTotal = BigDecimal.ZERO;
                BigInteger grandHashes = BigInteger.ZERO;
                int shardsCount = 0;

                ExtraBigShardPayload[] shards = {
                    req.shard01(), req.shard02(), req.shard03(), req.shard04(), req.shard05(),
                    req.shard06(), req.shard07(), req.shard08(), req.shard09(), req.shard10()
                };

                for (var shard : shards) {
                    if (shard != null) {
                        shardsCount++;
                        if (shard.shardAmount() != null) {
                            grandTotal = grandTotal.add(shard.shardAmount());
                        }
                        if (shard.shardTotalHashes() != null) {
                            grandHashes = grandHashes.add(shard.shardTotalHashes());
                        }
                    }
                }

                long seq = ID_SEQ.incrementAndGet();
                return new ExtraBigOrderResponse(
                    "ORD-XB-" + seq,
                    req.batchId(),
                    req.userId(),
                    "PROCESSED",
                    req.submittedAt() != null ? req.submittedAt() : Instant.now(),
                    grandTotal,
                    grandHashes,
                    shardsCount,
                    req.shard01(), req.shard02(), req.shard03(), req.shard04(), req.shard05(),
                    req.shard06(), req.shard07(), req.shard08(), req.shard09(), req.shard10()
                );
            }

            private ExtraBigShardPayload buildSampleShard(int idx, Instant now) {
                var items = new ArrayList<LargeOrderItem>(4);
                for (int i = 0; i < 4; i++) {
                    items.add(new LargeOrderItem(
                        "SKU-XB-" + idx + "-" + (100 + i),
                        "High-Density Compute Blade Node Cluster Unit " + idx + "." + i,
                        "Enterprise Silicon Systems Corp",
                        "Datacenter/Compute/Blades/Gen" + (i + 1),
                        "Multi-socket high-throughput compute node with hardware telemetry, ECC memory scrubbing, and redundant NVMe fabrics.",
                        2 + i,
                        new BigDecimal("38722.50"),
                        new BigDecimal("1200.00"),
                        new BigDecimal("0.20"),
                        new BigDecimal("7504.50"),
                        new BigDecimal("82549.50"),
                        4.85,
                        List.of(
                            new ItemAttribute("tdp", "350", "W", true),
                            new ItemAttribute("interconnect", "400GbE", "bps", true),
                            new ItemAttribute("compliance", "SOC2-TYPE2-2026", null, false)
                        ),
                        List.of(
                            new WarehouseAllocation("WH-NSK-01", "SIBERIA", 4, 6),
                            new WarehouseAllocation("WH-SPB-03", "NORTHWEST", 2, 24)
                        ),
                        List.of("DATACENTER", "HIGH_DENSITY", "VERIFIED")
                    ));
                }
            
                var packets = List.of(
                    new TelemetryPacket(
                        UUID.nameUUIDFromBytes(("pkt-a-" + idx).getBytes(StandardCharsets.UTF_8)),
                        100000L + idx,
                        (short) (10 + idx),
                        2,
                        "A",
                        true,
                        Boolean.TRUE,
                        ("sig-payload-shard-" + idx + "-alpha").chars().boxed().toList(),
                        url("https://koraframework.io" + idx),
                        Duration.ofMillis(145 + idx),
                        Duration.ofDays(365),
                        LocalDate.of(2026, 11, 28),
                        LocalTime.of(12, 30, idx),
                        LocalDateTime.of(2026, 11, 28, 12, 30, idx),
                        OffsetDateTime.of(2026, 11, 28, 12, 30, idx, 0, ZoneOffset.ofHours(7)),
                        ZonedDateTime.of(2026, 11, 28, 12, 30, idx, 0, ZoneId.of("Asia/Novosibirsk")),
                        new BigInteger("12345678901234567890123456789" + idx),
                        new BigDecimal("99.987654"),
                        Set.of("AVX512", "ZGC", "IO_URING", "ZERO_COPY"),
                        Map.of("temp_c", 42.5 + idx, "voltage_v", 12.04, "fan_rpm", 4800.0),
                        new GeoCoordinateNode(54.8471 + idx * 0.001, 83.1056 + idx * 0.001, 165.5f, 15, now, "RU-NSK-Z" + idx)
                    ),
                    new TelemetryPacket(
                        UUID.nameUUIDFromBytes(("pkt-b-" + idx).getBytes(StandardCharsets.UTF_8)),
                        200000L + idx,
                        (short) (20 + idx),
                        2,
                        "B",
                        false,
                        Boolean.FALSE,
                        ("sig-payload-shard-" + idx + "-beta").chars().boxed().toList(),
                        url("https://koraframework.io" + idx),
                        Duration.ofMillis(88 + idx),
                        Duration.ofDays(18),
                        LocalDate.of(2026, 11, 28),
                        LocalTime.of(13, 15, idx),
                        LocalDateTime.of(2026, 11, 28, 13, 15, idx),
                        OffsetDateTime.of(2026, 11, 28, 13, 15, idx, 0, ZoneOffset.UTC),
                        ZonedDateTime.of(2026, 11, 28, 13, 15, idx, 0, ZoneId.of("UTC")),
                        new BigInteger("98765432109876543210987654321" + idx),
                        new BigDecimal("100.000125"),
                        Set.of("TLS13", "FASTJSON2", "COMPACT_HEADERS"),
                        Map.of("temp_c", 39.1 + idx, "voltage_v", 11.98, "fan_rpm", 4200.0),
                        new GeoCoordinateNode(55.0302 + idx * 0.001, 82.9204 + idx * 0.001, 150.0f, 10, now, "RU-NSK-C" + idx)
                    )
                );
            
                var compliance = new NestedComplianceBlock(
                    "REG-2026-XBIG-" + idx,
                    "Automated cross-border telemetry and financial settlement compliance verification payload block.",
                    packets,
                    Map.of(
                        "allowedRegions", List.of("RU", "KZ", "BY", "AM"),
                        "requiredChecks", List.of("AML", "KYC", "SANCTIONS", "HASH_INTEGRITY"),
                        "fallbackRoutes", List.of("route-primary", "route-secondary")
                    )
                );
            
                return new ExtraBigShardPayload(
                    "SHARD-0" + idx,
                    UUID.nameUUIDFromBytes(("shard-trace-" + idx).getBytes(StandardCharsets.UTF_8)),
                    7000000000L + idx,
                    7000000000L + idx,
                    100 + idx,
                    100 + idx,
                    (short) idx,
                    (short) idx,
                    idx & 0x7F,
                    idx & 0x7F,
                    0.9995 + idx * 0.00001,
                    0.9995 + idx * 0.00001,
                    98.5f + idx,
                    98.5f + idx,
                    true,
                    Boolean.TRUE,
                    "S",
                    "S",
                    new BigDecimal("154890.00"),
                    new BigInteger("98765432109876543210987654321"),
                    now,
                    LocalDate.of(2026, 11, 28),
                    LocalTime.of(18, 0, 0),
                    LocalDateTime.of(2026, 11, 28, 18, 0, 0),
                    OffsetDateTime.of(2026, 11, 28, 18, 0, 0, 0, ZoneOffset.ofHours(7)),
                    ZonedDateTime.of(2026, 11, 28, 18, 0, 0, 0, ZoneId.of("Asia/Novosibirsk")),
                    Duration.ofMinutes(30),
                    Duration.ofMinutes(1),
                    url("https://enterprise-corp.ru" + idx),
                    ("binary-audit-blob-content-for-shard-" + idx).chars().boxed().toList(),
                    Set.of(1, 2, 4, 8, 16, 32),
                    List.of("rk.ingest.v2", "rk.shard." + idx, "rk.settlement.fast"),
                    Map.of("env", "benchmark", "shardIndex", String.valueOf(idx), "dc", "nsk-1"),
                    new CustomerProfile(
                        "Dmitry Sokolov " + idx, "d.sokolov" + idx + "@enterprise-corp.ru", "+7913555010" + idx,
                        "ENTERPRISE_VIP", 250000 + idx, Map.of("tier", "S1", "region", "SIBERIA")
                    ),
                    new DeliveryAddress(
                        "RU", "Novosibirsk", "Inzhenernaya", String.valueOf(10 + idx), "Suite " + idx,
                        "630090", 54.8581, 83.1112, "Datacenter loading dock #" + idx
                    ),
                    items,
                    compliance
                );
            }
            
            private static String url(String raw) {
                try {
                    return URI.create(raw).toURL().toString();
                } catch (Exception e) {
                    throw new IllegalArgumentException(e);
                }
            }
        }
        """.trimIndent()
    )

    // 3. КОНТРОЛЛЕР С 3 ОБОСОБЛЕННЫМИ ЭНДПОИНТАМИ (/bench/small, /bench/medium, /bench/large)
    val benchControllerFile = File("$targetDir/io/koraframework/benchmark/java/bench/IsolatedBenchmarkController.java")
    benchControllerFile.writeText(
        """
        package io.koraframework.benchmark.java.bench;

        import io.koraframework.benchmark.java.bench.dto.BenchmarkDtos.*;
        import io.koraframework.common.annotation.Component;
        import io.koraframework.http.common.HttpMethod;
        import io.koraframework.http.common.annotation.HttpRoute;
        import io.koraframework.http.common.annotation.Path;
        import io.koraframework.http.server.common.annotation.HttpController;
        import io.koraframework.json.common.annotation.Json;

        @HttpController
        @Component
        public class IsolatedBenchmarkController {

            private final IsolatedBenchmarkService service;

            public IsolatedBenchmarkController(IsolatedBenchmarkService service) {
                this.service = service;
            }

            // 1. SMALL (~100 bytes)
            @Json
            @HttpRoute(method = HttpMethod.GET, path = "/bench/small/{id}")
            public SmallOrderResponse getSmall(@Path String id) {
                return service.getSmallOrder(id);
            }

            @Json
            @HttpRoute(method = HttpMethod.POST, path = "/bench/small")
            public SmallOrderResponse postSmall(@Json SmallOrderRequest body) {
                return service.processSmallOrder(body);
            }

            // 2. MEDIUM (~1.5 - 2 KB)
            @Json
            @HttpRoute(method = HttpMethod.GET, path = "/bench/medium/{id}")
            public MediumOrderResponse getMedium(@Path String id) {
                return service.getMediumOrder(id);
            }

            @Json
            @HttpRoute(method = HttpMethod.POST, path = "/bench/medium")
            public MediumOrderResponse postMedium(@Json MediumOrderRequest body) {
                return service.processMediumOrder(body);
            }

            // 3. LARGE (~15 - 25 KB)
            @Json
            @HttpRoute(method = HttpMethod.GET, path = "/bench/large/{id}")
            public LargeOrderResponse getLarge(@Path String id) {
                return service.getLargeOrder(id);
            }

            @Json
            @HttpRoute(method = HttpMethod.POST, path = "/bench/large")
            public LargeOrderResponse postLarge(@Json LargeOrderRequest body) {
                return service.processLargeOrder(body);
            }

            // 4. EXTRABIG (~70 KB)
            @Json
            @HttpRoute(method = HttpMethod.GET, path = "/bench/extrabig/{id}")
            public ExtraBigOrderResponse getExtraBig(@Path String id) {
                return service.getExtraBigOrder(id);
            }

            @Json
            @HttpRoute(method = HttpMethod.POST, path = "/bench/extrabig")
            public ExtraBigOrderResponse postExtraBig(@Json ExtraBigOrderRequest body) {
                return service.processExtraBigOrder(body);
            }
        }
        """.trimIndent()
    )
}

// byte, char, Period, URI, URL, int[]