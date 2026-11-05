package io.koraframework.benchmark.java;

import com.alibaba.fastjson2.*;
import io.koraframework.application.graph.*;
import io.koraframework.benchmark.java.api.dto.*;
import io.koraframework.benchmark.java.api.dto.OrderDto1.*;
import io.koraframework.json.common.*;

import java.io.ByteArrayOutputStream;
import java.math.*;
import java.nio.charset.*;
import java.util.*;

import static java.lang.IO.println;

public final class JsonInMemoryBenchmark {

    public static void runBenchmark(ApplicationGraphDraw graphDraw) throws Exception {
        var graph = graphDraw.init();
        var readerCodec = (JsonReader<CreateOrderRequest1>) graph.get(AppGraph.holder0.component151);
        var writerCodec = (JsonWriter<OrderResponse1>) graph.get(AppGraph.holder0.component146);

        println("=== STARTING IN-MEMORY JACKSON CODEC BENCHMARK ===");

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
              "tags": ["benchmark", "jackson"]
            }
            """;
        byte[] jsonBytes = rawJson.getBytes(StandardCharsets.UTF_8);

        int warmupIterations = 250_000;
        int measurementIterations = 5_000_000;

        var testObject = new OrderResponse1(
            "122212",
            110266614641525L,
            new BigDecimal("123.56"),
            "FAILED"
        );

        System.out.println("Warming up JIT compiler for Jackson, please wait...");
        for (int i = 0; i < warmupIterations; i++) {
            // Jackson execution path for Read
            readerCodec.read(jsonBytes);

            // Jackson execution path for Write using native fast method
            writerCodec.toByteArray(testObject);
        }

        System.gc();
        Thread.sleep(1000);
        System.out.println("Warmup completed. Running Jackson measurement loops...");

        // READ PERFORMANCE
        long startTime = 0;
        long readTime = 0;
        double readRps = 0;

        for (int j = 0; j < 100; j++) {
            startTime = System.nanoTime();
            for (int i = 0; i < measurementIterations; i++) {
                var result = readerCodec.read(jsonBytes);
            }
            readTime = System.nanoTime() - startTime;
            readRps = measurementIterations / (readTime / 1_000_000_000.0);
        }

        long writeTime = 0;
        double writeRps = 0;
        // WRITE PERFORMANCE
        for (int j = 0; j < 100; j++) {
            startTime = System.nanoTime();
            for (int i = 0; i < measurementIterations; i++) {
                byte[] output = writerCodec.toByteArray(testObject);
            }
            writeTime = System.nanoTime() - startTime;
            writeRps = measurementIterations / (writeTime / 1_000_000_000.0);
        }

        /*
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
*/

        println("\n=================== JACKSON METRICS ===================");
        System.out.println("\n=================== BENCHMARK EXECUTION METRICS ===================");
        System.out.format(Locale.US, "Jackson Read Throughput : %,.2f operations/sec (Duration: %.3f s)%n", readRps, readTime / 1_000_000_000.0);
        System.out.format(Locale.US, "Jackson Write Throughput: %,.2f operations/sec (Duration: %.3f s)%n", writeRps, writeTime / 1_000_000_000.0);
        System.out.println("===================================================================\n");
        println("=========================================================\n");
    }
}
