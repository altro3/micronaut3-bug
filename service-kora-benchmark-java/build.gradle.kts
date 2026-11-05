import com.google.cloud.tools.jib.gradle.*
import org.gradle.api.tasks.testing.logging.*
import java.io.BufferedReader
import java.io.InputStreamReader

plugins {
    id("io.koraframework.kora-aot") version ("2.0.0-1-SNAPSHOT")
    application
    alias(libs.plugins.graal)
    alias(libs.plugins.jib)
    alias(libs.plugins.shadow)
}

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.4-x86_64"

dependencies {

    annotationProcessor(kora.koraframework.annotationProcessors)

    implementation("io.koraframework.config.test:service-kora-config-lib1:1.0.0-SNAPSHOT")
    implementation("io.koraframework.config.test:service-kora-config-lib2:1.0.0-SNAPSHOT")
    implementation(kora.koraframework.httpServerUndertow)
    implementation(kora.koraframework.loggingLogback)
    implementation(kora.koraframework.jsonCommon)
    implementation(kora.koraframework.validationModule)
    implementation(kora.koraframework.configCommon)
//    implementation(kora.koraframework.configHocon)
//    implementation(kora.koraframework.databaseJdbcPostgres)
//    implementation(kora.koraframework.databaseFlyway)
//    implementation(libs.flyway) {
//        exclude(group = "org.flywaydb", module = "flyway-database-cockroachdb")
//    }
//    implementation(kora.koraframework.cacheCaffeine)
//    implementation(kora.koraframework.openapiManagement)
//    implementation(kora.koraframework.resilientKora)
//    implementation(kora.koraframework.micrometerModule)
}

//tasks.koraAot {
//    configEnabled = false
//}

val isWindows = System.getProperty("os.name").contains("win", ignoreCase = true)

configurations.runtimeClasspath {
    exclude(group = "org.jspecify", module = "jspecify")
}

val jvmOpts = listOf(
    "-Xms16m",
    "-Xmx256m",
    "-Xss256k",
    "-XX:MaxMetaspaceSize=32m",
    "-XX:+UseG1GC",
    "-XX:+G1PeriodicGCInvokesConcurrent",
    "-XX:G1PeriodicGCInterval=1000",
    "-XX:MinHeapFreeRatio=10",
    "-XX:MaxHeapFreeRatio=20",
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
    "-XX:+DisableAttachMechanism",
)

val nativeRuntimeOpts = listOf(
    "-Xms64m",
    "-Xmx128m",
    "-Xss256k",
    "-Dfile.encoding=UTF-8",
    "-Dconfig.override_with_env_vars=true",

    // только для старой версии!
//    "-Dflyway.locations=filesystem:D:/Work/micronaut/micronaut3-bug/service-kora-benchmark-java/build/resources/main/db/migration"
)

tasks.named("nativeCompile") {
    dependsOn(tasks.named("processResources"))
}

graalvmNative {
    binaries {
        named("main") {
            imageName.set("service-kora")

            buildArgs.addAll(
                "-O4",
                "-Dfile.encoding=UTF-8",
                "-march=native",
                "-H:+Inline",
                "-H:+StaticExecutableWithDynamicLibC",

                // только для старой версии!
//                "--enable-url-protocols=https",
            )

            if (project.hasProperty("pgoMode")) {
                when (project.property("pgoMode").toString().lowercase()) {
                    "instrument" -> {
                        buildArgs.add("--pgo-instrument")
                        logger.lifecycle("PGO: Instrumentation mode enabled (--pgo-instrument)")
                    }
                    "optimize" -> {
                        val profileFile = project.layout.projectDirectory.file("default.iprof").asFile
                        if (profileFile.exists()) {
                            buildArgs.add("--pgo=${profileFile.absolutePath}")
                            logger.lifecycle("PGO: Optimization mode enabled using profile: ${profileFile.absolutePath}")
                        } else {
                            throw GradleException("PGO Error: Profile file default.iprof not found for 'optimize' mode!")
                        }
                    }
                    else -> logger.lifecycle("PGO: PGO disabled (unknown mode provided)")
                }
            } else {
                logger.lifecycle("PGO: Building without PGO (pgoMode property not provided)")
            }

//            if (!isWindows) {
                buildArgs.add("--gc=G1")
//            }

//            environmentVariables.put(
//                "INCLUDE", listOf(
//                    "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.51.36231\\include",
//                    "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\ucrt",
//                    "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\shared",
//                    "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\um"
//                ).joinToString(";")
//            )
//
//            environmentVariables.put(
//                "LIB", listOf(
//                    "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.51.36231\\lib\\x64",
//                    "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.26100.0\\ucrt\\x64",
//                    "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.26100.0\\um\\x64"
//                ).joinToString(";")
//            )
        }
    }
}

application {
    applicationName = "application"
    mainClass = "io.koraframework.benchmark.java.App"
    applicationDefaultJvmArgs = jvmOpts
}

tasks.register<Copy>("copyDependencies") {
    from(configurations.runtimeClasspath)
    into(layout.buildDirectory.dir("dependencies"))
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
    description = "Runs cold start benchmark for JIT (JAR) or GraalVM Native Image"

    val isNative = project.hasProperty("native")

    if (isNative) {
        dependsOn("nativeCompile")
    } else {
        dependsOn("shadowJar")
    }

    doLast {
        val runs = 100
        val rawResults = mutableListOf<Int>()
        val shouldGenerateCds = !isNative && project.hasProperty("generateCds") && project.property("generateCds").toString().toBoolean()

        val buildDirFile = project.layout.buildDirectory.asFile.get()
        val libsDir = if (isNative) File(buildDirFile, "native/nativeCompile") else File(buildDirFile, "libs")

        val exeExtension = if (isWindows) ".exe" else ""
        val executableName = if (isNative) "service-kora$exeExtension" else "service-kora.jar"

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
            command.addAll(
                listOf(
                    "-XX:ArchiveClassesAtExit=service.jsa",
                    "-Dkora.exit.after.init=true",
                    "-jar",
                    executableName
                )
            )

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
        println("STARTING ${if (isNative) "GRAALVM NATIVE" else "JVM JIT"} BENCHMARK ($runs RUNS)")
        println("==================================================")

        for (i in 1..runs) {
            print("Running iteration $i of $runs... ")

            val command = mutableListOf<String>()

            if (isNative) {
                val absoluteExePath = File(libsDir, executableName).absolutePath
                command.add(absoluteExePath)
                command.addAll(nativeRuntimeOpts)
            } else {
                command.add("java")
                command.addAll(jvmOpts)
                if (shouldGenerateCds) {
                    command.add("-XX:SharedArchiveFile=service.jsa")
                }
                command.addAll(listOf("-jar", executableName))
            }
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
            val sortedResults = rawResults.sorted()
            val reportName = if (isNative) "benchmark_native_results.csv" else "benchmark_raw_results.csv"
            val csvFile = File(buildDirFile, reportName)

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
            println("DISTRIBUTION HISTOGRAM (Baskets):")
            var countUnder20 = 0
            var count20to100 = 0
            var count100to150 = 0
            var count150to200 = 0
            var count200to250 = 0
            var count250to300 = 0
            var count300to350 = 0
            var count350to400 = 0
            var countOver400 = 0

            rawResults.forEach { time ->
                when {
                    time < 20 -> countUnder20++
                    time in 20..99 -> count20to100++
                    time in 100..149 -> count100to150++
                    time in 150..199 -> count150to200++
                    time in 200..249 -> count200to250++
                    time in 250..299 -> count250to300++
                    time in 300..349 -> count300to350++
                    time in 350..400 -> count350to400++
                    else -> countOver400++
                }
            }

            println("  < 20 ms    : $countUnder20 runs " + "*".repeat(minOf(30, (countUnder20 * 30 / runs))))
            println("  20-100 ms  : $count20to100 runs " + "*".repeat(minOf(30, (count20to100 * 30 / runs))))
            println("  100-150 ms : $count100to150 runs " + "*".repeat(minOf(30, (count100to150 * 30 / runs))))
            println("  150-200 ms : $count150to200 runs " + "*".repeat(minOf(30, (count150to200 * 30 / runs))))
            println("  200-250 ms : $count200to250 runs " + "*".repeat(minOf(30, (count200to250 * 30 / runs))))
            println("  250-300 ms : $count250to300 runs " + "*".repeat(minOf(30, (count250to300 * 30 / runs))))
            println("  300-350 ms : $count300to350 runs " + "*".repeat(minOf(30, (count300to350 * 30 / runs))))
            println("  350-400 ms : $count350to400 runs " + "*".repeat(minOf(30, (count350to400 * 30 / runs))))
            println("  > 400 ms   : $countOver400 runs " + "*".repeat(minOf(30, (countOver400 * 30 / runs))))
        } else {
            println("No data captured.")
        }
        println("==================================================")
    }
}
