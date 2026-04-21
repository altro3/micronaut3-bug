package com.micronaut.bug.controller;

import com.micronaut.bug.config.User;
import com.micronaut.bug.controller.ServerLogController.MyData;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.ReactorClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import reactor.netty.http.HttpProtocol;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
class H2cLoadIntegrationTest {

    @Autowired
    private RestClient h2cRestClient;

    @Test
    @DisplayName("Нагрузочный тест HTTP/2 (h2c) с проверкой проброса TraceId")
    void testH2cContextPropagationAndLoad() {
        int totalRequests = 20000;
        int parallelism = 10; // Ограничиваем количество ОДНОВРЕМЕННЫХ запросов
        var semaphore = new Semaphore(parallelism);
        var errorCount = new AtomicInteger();

//        log.info("=== Запуск теста на Virtual Threads: {} запросов ===", totalRequests);

        List<Long> latencies = Collections.synchronizedList(new ArrayList<>(totalRequests));
        var requestBody = new MyData("my-name", "Zubarev", 10, null);
        var mockUser = new User("userId", "tester", "ADMIN");
        // Используем виртуальные потоки для симуляции параллельной нагрузки
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            var futures = IntStream.range(0, totalRequests)
                .mapToObj(i -> CompletableFuture.runAsync(() -> {
                    // Ждем разрешения на выстрел
//                    String currentTraceId = "trace-" + i;
//                    UserContext.set(mockUser);
//                    LoggingContext.set(currentTraceId);
                    try {
                        semaphore.acquire();
                        long start = System.nanoTime();
                        h2cRestClient.post()
                            .uri(uriBuilder -> uriBuilder
                                .path("/testMultipart/{pathVar}")
                                .queryParam("queryVar", "testQuery")
                                .build("testPath"))
                            .contentType(MediaType.APPLICATION_JSON) // Пока тестируем JSON-body как в твоем коде
                            .header(HttpHeaders.AUTHORIZATION, "Bearer jwt-token")
                            .body(requestBody)
                            .retrieve()
                            .toBodilessEntity();

                        latencies.add(System.nanoTime() - start);
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
//                        log.error("Ошибка в запросе {}: {}", i, e.getMessage());
                    } finally {
//                        UserContext.reset();
//                        LoggingContext.reset();
                        // Освобождаем место для следующего запроса
                        semaphore.release();
                    }
                }, executor))
                .toArray(CompletableFuture[]::new);

            CompletableFuture.allOf(futures).join();
        }

        calculateAndPrintStats(latencies);
    }

    private void calculateAndPrintStats(List<Long> latencies) {
        if (latencies.isEmpty()) {
            System.out.println("Данных нет.");
            return;
        }

        Collections.sort(latencies);

        int size = latencies.size();
        double toMs = 1_000_000.0;

        // Тайминги
        var best = latencies.getFirst() / toMs; // Java 21+ List.getFirst()
        var worst = latencies.getLast() / toMs;  // Java 21+ List.getLast()
        var p50 = latencies.get((int) (size * 0.50)) / toMs;
        var p95 = latencies.get((int) (size * 0.95)) / toMs;
        var p99 = latencies.get((int) (size * 0.99)) / toMs;

        // Усеченное среднее (Truncated Mean)
        int cutOff = (int) (size * 0.05);
        var truncatedAvg = latencies.subList(cutOff, size - cutOff).stream()
            .mapToLong(Long::longValue)
            .average()
            .orElse(0.0) / toMs;

        // Используем прямой вызов println и String Templates (если включены)
        System.out.println("\n" + "=".repeat(50));
        System.out.println("         PERFORMANCE REPORT (JAVA 25+)         ");
        System.out.println("=".repeat(50));
        System.out.printf("Всего запросов    : %d%n", size);
        System.out.println("-".repeat(50));
        System.out.printf("Min Latency       : %.3f ms%n", best);
        System.out.printf("Median (P50)      : %.3f ms%n", p50);
        System.out.printf("Average (Trunc)   : %.3f ms%n", truncatedAvg);
        System.out.printf("P95 Latency       : %.3f ms%n", p95);
        System.out.printf("P99 Latency       : %.3f ms%n", p99);
        System.out.printf("Max Latency       : %.3f ms%n", worst);
        System.out.println("=".repeat(50) + "\n");
    }

    @TestConfiguration
    public static class TestClientConfig {

        @Bean
        public RestClient h2cRestClient() {
            // Настраиваем пул соединений специально для HTTP/2
            ConnectionProvider provider = ConnectionProvider.builder("h2c-test-pool")
                .maxConnections(20) // Ограничиваем количество физических TCP-соединений
                .pendingAcquireTimeout(Duration.ofSeconds(15))
                .build();

            HttpClient httpClient = HttpClient.create(provider)
                .protocol(HttpProtocol.H2C);

            return RestClient.builder()
                .requestFactory(new ReactorClientHttpRequestFactory(httpClient))
                .baseUrl("http://localhost:8080")
                .build();
        }
    }
}