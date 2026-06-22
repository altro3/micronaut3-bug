import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent

plugins {
    alias(libs.plugins.jib)
    alias(libs.plugins.spring.boot)
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.3-x86_64"

dependencies {

    kapt(spring.spring.springBootConfigurationProcessor)

    implementation(spring.spring.springBootStarterWeb)
    implementation(spring.spring.springBootStarterLog4j2)
    implementation(spring.spring.springBootStarterValidation)
    implementation(spring.spring.springBootStarterActuator)
    implementation(spring.spring.springBootStarterDataJdbc)
    implementation(spring.flywaydb.flywayCore)
    implementation(spring.flywaydb.flywayDatabasePostgresql)
    implementation(spring.postgresql.postgresql)
    implementation(spring.caffeine.caffeine)
    implementation(spring.micrometer.micrometerRegistryOtlp)
    implementation(springAi.spring.springAiStarterModelOpenai)
    implementation(springAi.spring.springAiStarterVectorStoreQdrant)
    implementation(springAi.spring.springAiStarterMcpClient)
    implementation(springAi.spring.springAiVectorStoreAdvisor)
    implementation(springAi.spring.mcpSpringWebmvc)
    implementation(coroutines.kotlinx.kotlinxCoroutinesCoreJvm)
    implementation(coroutines.kotlinx.kotlinxCoroutinesSlf4j)
    implementation(kt.kotlin.kotlinReflect)
    implementation(libs.kotlin.logging)
    implementation(projects.mycommon)
    implementation(projects.myutil)
    implementation(projects.mytracelog)
    implementation(projects.myhttp)
    implementation(projects.myflyway)
}
configurations.all {
    exclude(group = "org.springframework.boot", module = "spring-boot-starter-logging")
}

kotlin {
    jvmToolchain(25)
    compilerOptions {
        freeCompilerArgs.addAll("-Xjsr305=strict", "-Xemit-jvm-type-annotations", "-Xannotation-default-target=param-property", "-jvm-default=enable")
        javaParameters = true
    }
}

jib {
    from { image = jreImage }
    to { image = "localhost:5000/myorchestrator-spring:latest" }
    container {
        jvmFlags = listOf(
            "-XX:+UseG1GC",
            "-XX:+UseStringDeduplication",
            "-XX:MaxRAMPercentage=75.0",
            "-Dfile.encoding=UTF-8",
        )
    }
    setAllowInsecureRegistries(true)
}

tasks.bootJar {
    enabled = false
    layered {
        enabled = true
    }
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
