import org.gradle.api.tasks.testing.logging.TestLogEvent
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.springframework.boot") version "3.5.13"
    id("io.spring.dependency-management") version "1.1.7"
    kotlin("jvm") version "2.3.20"
    kotlin("plugin.spring") version "2.3.20"
    kotlin("kapt") version "2.3.20" // Добавили плагин здесь
}

val ver = mapOf(
    "kotlin" to "2.3.20",
    "springBoot" to "3.5.13",
)

val jreImage = "bellsoft/liberica-openjre-alpine:25.0.2"

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
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.01")

    // Tracing & Logs
    implementation("io.micrometer:micrometer-tracing")
    implementation("io.micrometer:context-propagation")
    implementation("ch.qos.logback:logback-classic")

    // Testing
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
    testImplementation("io.projectreactor.netty:reactor-netty-http")
}

tasks.withType<KotlinCompile>().configureEach {
    compilerOptions {
        freeCompilerArgs.addAll("-Xjsr305=strict", "-java-parameters")
    }
}

configurations.all {
    resolutionStrategy {
        cacheDynamicVersionsFor(0, "minutes")
        cacheChangingModulesFor(0, "minutes")
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
        "-XX:+UseG1GC",
        "-XX:+UseStringDeduplication",
        "-Dfile.encoding=UTF-8"
    )
}
