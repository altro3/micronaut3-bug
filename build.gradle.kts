import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    alias(libs.plugins.spring.boot)
    alias(libs.plugins.jib)
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

val ver = mapOf(
    "kotlin" to "2.3.21",
    "springBoot" to "3.5.14",
)

val jreImage = "bellsoft/liberica-openjre-alpine:21.0.11-x86_64"

kotlin {
    jvmToolchain(21)
}

repositories {
    mavenCentral()
}

dependencies {

    kapt(spring.spring.springBootConfigurationProcessor)

    implementation(spring.spring.springBootStarterWeb)
    implementation(spring.spring.springBootStarterLogging)
    implementation(spring.spring.springBootStarterJson)
    implementation(spring.spring.springBootStarterValidation)
    implementation(spring.spring.springBootStarterActuator)
    implementation(spring.spring.springRetry)
    implementation(spring.projectreactor.reactorNettyHttp)
    implementation(spring.jackson.jacksonModuleKotlin)
    implementation(spring.jackson.jacksonModuleBlackbird)
    implementation(spring.logback.logbackClassic)
    implementation(coroutines.kotlinx.kotlinxCoroutinesCoreJvm)
    implementation(coroutines.kotlinx.kotlinxCoroutinesSlf4j)
    implementation(kot.kotlin.kotlinReflect)
    implementation(libs.kotlin.logging)
    implementation(libs.loki.logback)
    implementation(libs.loki.protobuf)
    implementation(libs.protobuf.java)
    implementation(libs.otel.proto)
    implementation(libs.wiremock)

    testImplementation(spring.spring.springBootStarterTest)
    testImplementation(kot.kotlin.kotlinTestJunit5)
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
