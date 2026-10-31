import com.google.cloud.tools.jib.gradle.*
import org.gradle.api.tasks.testing.logging.*
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.roundToInt

plugins {
    id("io.koraframework.config-aot") version("2.0.0-1-SNAPSHOT")
    application
    alias(libs.plugins.jib)
    alias(libs.plugins.shadow)
}

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.4-x86_64"

dependencies {

    annotationProcessor(kora.koraframework.annotationProcessors)

    implementation("io.koraframework.config.test:service-kora-config-lib1:1.0.0-SNAPSHOT")
    implementation("io.koraframework.config.test:service-kora-config-lib2:1.0.0-SNAPSHOT")
    implementation(kora.koraframework.configCommon)
//    implementation(kora.koraframework.configHocon)
    implementation(kora.koraframework.loggingLogback)
    implementation(kora.koraframework.jsonCommon)
    implementation(kora.koraframework.validationModule)
    implementation(kora.koraframework.databaseJdbcPostgres)
    implementation(kora.koraframework.databaseFlyway)
    implementation(libs.flyway)
    implementation(kora.koraframework.httpServerUndertow)
//    implementation(kora.koraframework.cacheCaffeine)
//    implementation(kora.koraframework.openapiManagement)
//    implementation(kora.koraframework.resilientKora)
//    implementation(kora.koraframework.micrometerModule)
}

val jvmOpts = listOf(
    "-Xms512m",
    "-Xmx512m",
    "-Xss256k",
    "-XX:MaxMetaspaceSize=96m",
    "-XX:+UseG1GC",
    "-XX:MaxGCPauseMillis=50",
    "-XX:+G1PeriodicGCInvokesConcurrent",
    "-XX:+UseStringDeduplication",
    "-XX:+UnlockExperimentalVMOptions",
    "-XX:+UseCompactObjectHeaders",
    "-XX:+EliminateAllocations",
    "-Dfile.encoding=UTF-8",
    "-Dconfig.override_with_env_vars=true",
    "-Dcom.sun.management.jmxremote=false",
    "-Dcom.sun.management.jmxremote.ssl=false",
    "-Dcom.sun.management.jmxremote.port=0",
    "-Dcom.sun.management.jmxremote.authenticate=false",
    "-Djava.rmi.server.hostname=127.0.0.1",
//    "-XX:+DisableAttachMechanism"
)

application {
    applicationName = "application"
    mainClass = "io.koraframework.benchmark.java.App"
    applicationDefaultJvmArgs = jvmOpts
}

jib {
    from { image = jreImage }
    to { image = "localhost:5000/micronaut3-bug:latest" }
    container {
        jvmFlags = listOf("-XX:MaxRAMPercentage=75.0") + jvmOpts
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

configurations.all {
    resolutionStrategy.cacheChangingModulesFor(0, "seconds")
    resolutionStrategy.cacheDynamicVersionsFor(0, "seconds")
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

val shadowFlywayDir = File(project.layout.buildDirectory.asFile.get(), "tmp/shadowServices")

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

        val shouldGenerateCds = project.hasProperty("generateCds") && project.property("generateCds").toString().toBoolean()


        val buildDirFile = project.layout.buildDirectory.asFile.get()
        val libsDir = File(buildDirFile, "libs")
        val jarName = "service-kora.jar"

        if (shouldGenerateCds) {
            println("==================================================")
            println("GENERATING CDS ARCHIVE (ONE-PASS TRAINING RUN)...")
            println("==================================================")

            val cdaArchiveFile = File(libsDir, "service.jsa")
            if (cdaArchiveFile.exists()) {
                cdaArchiveFile.delete()
            }

            val command = mutableListOf("java")
            command.addAll(jvmOpts)
            command.addAll(listOf(
                "-XX:ArchiveClassesAtExit=service.jsa",
                "-Dkora.exit.after.init=true",
                "-jar",
                jarName
            ))

            val pb = ProcessBuilder(command)
            pb.directory(libsDir)
            pb.redirectErrorStream(true)
            val process = pb.start()

            BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    println("[CDS-DUMP-LOG]: $line")
                }
            }

            process.waitFor()
            println("CDS Phase finished. Checking archive file...")

            if (cdaArchiveFile.exists() && cdaArchiveFile.length() > 0) {
                println("SUCCESS: service.jsa generated successfully (${cdaArchiveFile.length()} bytes).")
            } else {
                println("ERROR: service.jsa was not created.")
            }

            println("Starting benchmark iterations...\n")
            Thread.sleep(1500)
        }

        println("==================================================")
        println("STARTING BENCHMARK (COLLECTING ALL RAW DATA)")
        println("==================================================")

        for (i in 1..runs) {
            print("Running iteration $i of $runs... ")

            val command = mutableListOf("java")
            command.addAll(jvmOpts)
            if (shouldGenerateCds) {
                command.add("-XX:SharedArchiveFile=service.jsa")
            }
            command.addAll(listOf("-jar", jarName))

            val pb = ProcessBuilder(command)
            pb.directory(libsDir)
            pb.redirectErrorStream(true)
            val process = pb.start()

            var startupTime: Int? = null
            val fullLog = StringBuilder()

            BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    fullLog.append(line).append("\n")
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

            if (startupTime != null) {
                rawResults.add(startupTime)
                println("SUCCESS: $startupTime ms")
            } else {
                println("FAILED")
                println("\n--- [PROCESS ERROR LOG START] ---")
                println(fullLog.toString().trim())
                println("--- [PROCESS ERROR LOG END] ---\n")
                println("Benchmark aborted due to process initialization failure.")
                return@doLast
            }

            Thread.sleep(100)
        }

        println("\n==================================================")
        println("EXTENDED BENCHMARK REPORT")
        println("==================================================")

        if (rawResults.isNotEmpty()) {
            // 1. Сортируем тайминги по возрастанию
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

            val calculateTrimmedMean = { percentile: Int ->
                val tailPercent = (100.0 - percentile) / 2.0 / 100.0
                val trimCount = (sortedResults.size * tailPercent).toInt()
                val trimmed = sortedResults.subList(trimCount, sortedResults.size - trimCount)
                if (trimmed.isNotEmpty()) trimmed.average().toInt() else 0
            }

            val avg50 = calculateTrimmedMean(50)
            val avg90 = calculateTrimmedMean(90)
            val avg98 = calculateTrimmedMean(98)

            println("SLA & TRIMMED MEANS (ALL $runs RUNS):")
            println("Absolute Min Time       : $absoluteMin ms")
            println("Trimmed Mean 50% (p50)  : $avg50 ms")
            println("Trimmed Mean 90% (p90)  : $avg90 ms")
            println("Trimmed Mean 98% (p98)  : $avg98 ms")
            println("Absolute Max Time       : $absoluteMax ms")
            println("--------------------------------------------------")

            println("DISTRIBUTION HISTOGRAM (Baskets):")
            var countUnder400 = 0
            var count400to500 = 0
            var count500to1000 = 0
            var countOver1000 = 0

            rawResults.forEach { time ->
                when {
                    time < 400 -> countUnder400++
                    time in 400..500 -> count400to500++
                    time in 501..999 -> count500to1000++
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
