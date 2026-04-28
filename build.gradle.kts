import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.springframework.boot") version "3.5.13"
    id("io.spring.dependency-management") version "1.1.7"
    id("com.google.cloud.tools.jib") version "3.5.3"
    kotlin("jvm") version "2.3.20"
    kotlin("plugin.spring") version "2.3.20"
    kotlin("kapt") version "2.3.20" // Добавили плагин здесь
}

val ver = mapOf(
    "kotlin" to "2.3.20",
    "springBoot" to "3.5.13",
)

val jreImage = "bellsoft/liberica-openjre-alpine:21.0.11-x86_64"

kotlin {
    jvmToolchain(21)
}

repositories {
    mavenCentral()
}

dependencies {

    // Configuration Processor (для @ConfigurationProperties)
    kapt("org.springframework.boot:spring-boot-configuration-processor")

    implementation(platform("org.springframework.boot:spring-boot-dependencies:${ver["springBoot"]}"))
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-aop")
    implementation("org.springframework.boot:spring-boot-starter-logging")
    implementation("org.springframework.boot:spring-boot-starter-json")
    implementation("org.springframework.boot:spring-boot-starter-validation")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.retry:spring-retry")
    implementation("io.projectreactor.netty:reactor-netty-http")

    // Kotlin Essential
    implementation("org.jetbrains.kotlin:kotlin-reflect")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin")

    // Tracing & Logs
    implementation("ch.qos.logback:logback-classic")
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.01")
    implementation("com.github.loki4j:loki-logback-appender:2.0.3")
    implementation("com.github.loki4j:loki-protobuf:0.0.2_pb4.33.0")
    implementation("com.google.protobuf:protobuf-java:4.34.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-slf4j:1.10.2")
    implementation("io.opentelemetry.proto:opentelemetry-proto:1.10.0-alpha")

    implementation("org.wiremock:wiremock-standalone:3.13.2")

    // Testing
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
    testImplementation("io.projectreactor.netty:reactor-netty-http")
}

configurations.all {
    resolutionStrategy {
        cacheDynamicVersionsFor(0, "minutes")
        cacheChangingModulesFor(0, "minutes")
    }
}

tasks.withType<KotlinCompile>().configureEach {
    compilerOptions {
        freeCompilerArgs.addAll("-Xjsr305=strict", "-java-parameters", "-Xemit-jvm-type-annotations", "-Xannotation-default-target=param-property")
    }
}

tasks.test {
    useJUnitPlatform()

    testLogging {
        events(TestLogEvent.PASSED, TestLogEvent.SKIPPED, TestLogEvent.FAILED)
        showStandardStreams = true
        showExceptions = true
        showStackTraces = true
    }

    outputs.upToDateWhen { false }

    minHeapSize = "2g"
    maxHeapSize = "4g"

    jvmArgs(
        "-XX:MaxMetaspaceSize=512m",
        "-XX:+EnableDynamicAgentLoading",
        "-XX:+UseG1GC",
        "-XX:+UseStringDeduplication",
        "-Dfile.encoding=UTF-8"
    )
}

tasks.bootJar {
    layered {
        enabled = true
    }
}

tasks.withType<JibTask> {
    notCompatibleWithConfigurationCache("Jib does not support the Gradle configuration cache yet")
}

jib {
    from { image = jreImage }
    to {
        image = "localhost:5000/micronaut3-bug:latest"
    }
    container {
        jvmFlags = listOf(
            "-XX:+UseG1GC",
            "-XX:+UseStringDeduplication",
            "-Dfile.encoding=UTF-8"
        )
    }
    setAllowInsecureRegistries(true)
}
