package io.koraframework.benchmark.java;

import com.alibaba.fastjson2.*;
import io.koraframework.application.graph.*;
import io.koraframework.benchmark.java.api.dto.*;
import io.koraframework.benchmark.java.api.dto.OrderDto1.*;
import io.koraframework.json.common.*;

import java.nio.charset.*;
import java.util.*;

import static java.lang.IO.println;

public final class JsonInMemoryBenchmark {

    public static void runBenchmark(ApplicationGraphDraw graphDraw) throws Exception {
        var graph = graphDraw.init();
        var codec = graph.get(AppGraph.Partition0.component159);
        var readerCodec = (JsonReader<CreateOrderRequest1>) codec;
        var writerCodec = (JsonWriter<CreateOrderRequest1>) codec;

        println("=== STARTING IN-MEMORY FASTJSON2 CODEC BENCHMARK ===");

        String rawJson = """
            {
              "userId": 110266614641525,
              "userEmail": "user@koraframework.io",
              "promoCode": "PROMO2026",
              "items": [
                {
                  "typeCode": "GOODS",
                  "quantity": 5
                }
              ],
              "tags": ["benchmark", "fastjson2"]
            }
            """;
        byte[] jsonBytes = rawJson.getBytes(StandardCharsets.UTF_8);

        var testObject = new CreateOrderRequest1(
            110266614641525L,
            "user@koraframework.io",
            "PROMO2026",
            List.of(new OrderDto1.ItemRequest1("GOODS", 5)),
            List.of("benchmark", "fastjson2")
        );

        int warmupIterations = 250_000;
        int measurementIterations = 5_000_000;

        println("Warming up JIT compiler for Fastjson2, please wait...");
        for (int i = 0; i < warmupIterations; i++) {
            try (var reader = JSONReader.of(jsonBytes, 0, jsonBytes.length)) {
                readerCodec.read(reader);
            }
            try (var writer = JSONWriter.ofUTF8()) {
                writerCodec.write(writer, testObject);
            }
        }

        System.gc();
        Thread.sleep(1000);
        println("Warmup completed. Running Fastjson2 measurement loops...");

        // READ PERFORMANCE
        long startTime = System.nanoTime();
        for (int i = 0; i < measurementIterations; i++) {
            try (var reader = JSONReader.of(jsonBytes, 0, jsonBytes.length)) {
                var result = readerCodec.read(reader);
            }
        }
        long readTime = System.nanoTime() - startTime;
        double readRps = measurementIterations / (readTime / 1_000_000_000.0);

        // WRITE PERFORMANCE
        startTime = System.nanoTime();
        for (int i = 0; i < measurementIterations; i++) {
            try (var writer = JSONWriter.ofUTF8()) {
                writerCodec.write(writer, testObject);
                byte[] output = writer.getBytes();
            }
        }
        long writeTime = System.nanoTime() - startTime;
        double writeRps = measurementIterations / (writeTime / 1_000_000_000.0);

        println("\n=================== FASTJSON2 METRICS ===================");
        System.out.println("\n=================== BENCHMARK EXECUTION METRICS ===================");
        System.out.format(java.util.Locale.US, "Fastjson2 Read Throughput : %,.2f operations/sec (Duration: %.3f s)%n", readRps, readTime / 1_000_000_000.0);
        System.out.format(java.util.Locale.US, "Fastjson2 Write Throughput: %,.2f operations/sec (Duration: %.3f s)%n", writeRps, writeTime / 1_000_000_000.0);
        System.out.println("===================================================================\n");
        println("=========================================================\n");
    }
}
