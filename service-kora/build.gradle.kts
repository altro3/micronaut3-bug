import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent

plugins {
    application
    alias(libs.plugins.jib)
    alias(libs.plugins.ksp)
    alias(libs.plugins.kotlin.jvm)
//    alias(libs.plugins.kotlin.kapt)
}

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.4-x86_64"

dependencies {
    ksp(kora.koraframework.symbolProcessors)

    implementation(kora.koraframework.httpServerUndertow)
    implementation(kora.koraframework.jsonCommon)
    implementation(kora.koraframework.loggingLogback)
    implementation(kora.koraframework.configHocon)
    implementation(kora.koraframework.databaseJdbcPostgres)
    implementation(kora.koraframework.cacheCaffeine)
    implementation(kora.koraframework.resilientKora)
    implementation(kora.koraframework.validationModule)
    implementation(kora.koraframework.mapstructKspExtension)
    implementation(kora.koraframework.micrometerModule)
    implementation(kora.koraframework.databaseFlyway)
    implementation(libs.kotlin.logging)
}

application {
    applicationName = "application"
    mainClass = "com.altro.servicekora.AppKt"
    applicationDefaultJvmArgs = listOf(
        "-XX:+UseG1GC",
        "-XX:+UseStringDeduplication",
        "-Dfile.encoding=UTF-8",
        "-Dconfig.override_with_env_vars=true",
    )
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
        jvmFlags = listOf(
            "-XX:+UseG1GC",
            "-XX:+UseStringDeduplication",
            "-XX:MaxRAMPercentage=75.0",
            "-Dfile.encoding=UTF-8",
            "-Dconfig.override_with_env_vars=true",
//            "-Dspring.aot.enabled=true",
        )
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
        "-Dfile.encoding=UTF-8",
    )
}
