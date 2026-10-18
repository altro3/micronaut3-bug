import com.google.cloud.tools.jib.gradle.*
import org.gradle.api.tasks.testing.logging.*
import java.io.BufferedReader
import java.io.InputStreamReader

plugins {
    application
    alias(libs.plugins.jib)
    alias(libs.plugins.ksp)
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.kapt)
    alias(libs.plugins.shadow)
}

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.4-x86_64"

dependencies {

    ksp(kora.koraframework.symbolProcessors)

    implementation(kora.koraframework.httpServerUndertow)
    implementation(kora.koraframework.configHocon)
    implementation(kora.koraframework.jsonCommon)
    implementation(kora.koraframework.loggingLogback)
    implementation(kora.koraframework.databaseFlyway)
    implementation(kora.koraframework.databaseJdbcPostgres)
    implementation(kora.koraframework.cacheCaffeine)
    implementation(kora.koraframework.openapiManagement)
    implementation(kora.koraframework.resilientKora)
    implementation(kora.koraframework.validationModule)
    implementation(kora.koraframework.micrometerModule)
    implementation(libs.flyway)
    implementation(libs.kotlin.logging)
}

val modernJvmOptimizations = listOf(
    "-Xms128m",
    "-Xmx256m",
    "-XX:MaxMetaspaceSize=64m",
    "-XX:+UseG1GC",
    "-XX:+UseStringDeduplication",
    "-XX:+UnlockExperimentalVMOptions",
    "-XX:+UseCompactObjectHeaders",
    "-XX:+EliminateAllocations",
    "-XX:+UseNUMA",
    "-Dfile.encoding=UTF-8",
    "-Dconfig.override_with_env_vars=true"
)

application {
    applicationName = "application"
    mainClass = "io.koraframework.benchmark.AppKt"
    applicationDefaultJvmArgs = modernJvmOptimizations
}

kotlin {
    jvmToolchain(25)
    compilerOptions {
        freeCompilerArgs.addAll("-Xjsr305=strict", "-Xemit-jvm-type-annotations", "-jvm-default=enable")
        javaParameters = true
    }
    sourceSets.main { kotlin.srcDir("build/generated/ksp/main/kotlin") }
    sourceSets.test { kotlin.srcDir("build/generated/ksp/test/kotlin") }
}

jib {
    from { image = jreImage }
    to { image = "localhost:5000/micronaut3-bug:latest" }
    container {
        jvmFlags = listOf("-XX:MaxRAMPercentage=75.0") + modernJvmOptimizations
    }

    extraDirectories {
        paths {
            path {
                setFrom("build/classes/java/aot")
                into = "/app/classes"
            }
            path {
                setFrom("build/generated/aotResources")
                into = "/app/resources"
            }
        }
    }

    setAllowInsecureRegistries(true)
}

tasks.withType<JibTask> {
    notCompatibleWithConfigurationCache("Jib does not support the Gradle configuration cache yet")
}

tasks.test {
    useJUnitPlatform()

    testLogging {
        events(TestLogEvent.PASSED, TestLogEvent.SKIPPED, TestLogEvent.FAILED)
        showExceptions = true
        showStackTraces = true
    }

    maxParallelForks = 4
    minHeapSize = "256m"
    maxHeapSize = "4g"

    jvmArgs(
        "-XX:MaxMetaspaceSize=384m",
        "-XX:+UseParallelGC",
        "-XX:+UnlockExperimentalVMOptions",
        "-XX:+UseCompactObjectHeaders",
        "-XX:+UseStringDeduplication",
        "-Dfile.encoding=UTF-8",
    )
}

tasks.shadowJar {
    archiveBaseName.set("service-kora")
    archiveClassifier.set("")
    mergeServiceFiles()
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}

tasks.distZip { enabled = false }
tasks.distTar { enabled = false }
tasks.startScripts { enabled = false }

tasks.named("shadowDistZip") { enabled = false }
tasks.named("shadowDistTar") { enabled = false }
tasks.named("startShadowScripts") { enabled = false }

tasks.register("benchmark") {
    group = "verification"
    description = "Runs cold start benchmark and saves raw data for analysis"

    dependsOn("shadowJar")

    doLast {
        val runs = 100
        val rawResults = mutableListOf<Int>()

        val jvmOpts = listOf(
            "-XX:+UseG1GC",
            "-XX:+UseStringDeduplication",
            "-XX:+UnlockExperimentalVMOptions",
            "-XX:+UseCompactObjectHeaders",
            "-XX:+EliminateAllocations",
            "-XX:+UseNUMA",
            "-Dfile.encoding=UTF-8",
            "-Dconfig.override_with_env_vars=true"
        )

        val shadowTask = tasks.findByName("shadowJar")
        val buildDirFile = project.layout.buildDirectory.asFile.get()

        val jarPath = if (shadowTask != null && shadowTask.hasProperty("archiveFile")) {
            val archiveFileProp = shadowTask.property("archiveFile") as? RegularFileProperty
            archiveFileProp?.get()?.asFile?.absolutePath ?: File(buildDirFile, "libs/service-kora.jar").absolutePath
        } else {
            File(buildDirFile, "libs/service-kora.jar").absolutePath
        }

        println("==================================================")
        println("STARTING BENCHMARK (COLLECTING ALL RAW DATA)")
        println("==================================================")

        for (i in 1..runs) {
            print("Running iteration $i of $runs... ")

            val command = mutableListOf("java")
            command.addAll(jvmOpts)
            command.addAll(listOf("-jar", jarPath))

            val pb = ProcessBuilder(command)
            pb.directory(project.rootDir)
            pb.redirectErrorStream(true)
            val process = pb.start()

            var startupTime: Int? = null

            BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    if (line!!.contains("Application initialized in")) {
                        val match = Regex("Application initialized in (\\d+)ms").find(line)
                        if (match != null) {
                            startupTime = match.groupValues[1].toInt()
                            process.destroyForcibly()
                            break
                        }
                    }
                }
            }

            process.waitFor()

            if (startupTime != null) {
                rawResults.add(startupTime)
                println("SUCCESS: $startupTime ms")
            } else {
                println("FAILED")
            }

            Thread.sleep(100)
        }

        println("\n==================================================")
        println("EXTENDED BENCHMARK REPORT")
        println("==================================================")

        if (rawResults.isNotEmpty()) {
            val sortedResults = rawResults.sorted()

            val csvFile = File(buildDirFile, "benchmark_raw_results.csv")
            csvFile.writeText("Iteration,StartupTimeMs\n")
            rawResults.forEachIndexed { index, time ->
                csvFile.appendText("${index + 1},$time\n")
            }
            println("Raw data exported to: ${csvFile.absolutePath}")
            println("--------------------------------------------------")

            val absoluteMin = sortedResults.first()
            val absoluteMax = sortedResults.last()

            val p50 = sortedResults[(sortedResults.size * 0.50).toInt()]
            val p95 = sortedResults[(sortedResults.size * 0.95).toInt()]
            val p99 = sortedResults[(sortedResults.size * 0.99).toInt()]

            val trimCount = (sortedResults.size * 0.05).toInt()
            val trimmedResults = sortedResults.subList(trimCount, sortedResults.size - trimCount)
            val avgTime = if (trimmedResults.isNotEmpty()) trimmedResults.average().toInt() else 0

            println("SLA & PERCENTILES (ALL $runs RUNS):")
            println("Absolute Min Time     : $absoluteMin ms")
            println("Median (p50)          : $p50 ms")
            println("Trimmed Mean (Avg 5%) : $avgTime ms")
            println("95th Percentile (p95) : $p95 ms")
            println("99th Percentile (p99) : $p99 ms")
            println("Absolute Max Time     : $absoluteMax ms")
            println("--------------------------------------------------")

            println("DISTRIBUTION HISTOGRAM (Baskets):")
            var countUnder400 = 0
            var count400to500 = 0
            var count500to1000 = 0
            var countOver1000 = 0

            rawResults.forEach { time ->
                when {
                    time < 400 -> countUnder400++
                    time in 400..499 -> count400to500++
                    time in 500..999 -> count500to1000++
                    else -> countOver1000++
                }
            }

            println("  < 400 ms   : $countUnder400 runs " + "*".repeat(minOf(30, (countUnder400 * 30 / runs))))
            println("  400-500 ms : $count400to500 runs " + "*".repeat(minOf(30, (count400to500 * 30 / runs))))
            println("  500-1000 ms: $count500to1000 runs " + "*".repeat(minOf(30, (count500to1000 * 30 / runs))))
            println("  > 1000 ms  : $countOver1000 runs " + "*".repeat(minOf(30, (countOver1000 * 30 / runs))))
        } else {
            println("No data captured.")
        }
        println("==================================================")
    }
}

