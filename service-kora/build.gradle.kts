import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent

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
    implementation(libs.flyway)
    implementation(kora.koraframework.databaseJdbcPostgres)
    implementation(kora.koraframework.cacheCaffeine)
    implementation(kora.koraframework.openapiManagement)
    implementation(kora.koraframework.resilientKora)
    implementation(kora.koraframework.validationModule)
    implementation(kora.koraframework.micrometerModule)
    implementation(libs.kotlin.logging)
}

val modernJvmOptimizations = listOf(
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
    "-Djava.rmi.server.hostname=127.0.0.1"
)

application {
    applicationName = "application"
    mainClass = "com.altro.servicekora.AppKt"
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

tasks.shadowJar {
    archiveBaseName.set("service-kora-benchmark")
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