import com.google.cloud.tools.jib.gradle.JibTask
import org.gradle.api.tasks.testing.logging.TestLogEvent
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    alias(libs.plugins.spring.boot)
//    alias(libs.plugins.spring.boot.aot)
    alias(libs.plugins.jib)
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
//    alias(libs.plugins.jmh)
}

val jreImage = "bellsoft/liberica-openjre-alpine:21.0.11-x86_64"

kotlin {
    jvmToolchain(21)
}

repositories {
    mavenCentral()
}

dependencies {

//    kaptJmh("org.openjdk.jmh:jmh-generator-annprocess:1.37")
//    jmh("org.openjdk.jmh:jmh-core:1.37")

    kapt(spring.spring.springBootConfigurationProcessor)
    kapt(spring.log4j.log4jCore)

    implementation(spring.spring.springBootStarterWeb) {
        exclude(group = "org.apache.tomcat.embed", module = "tomcat-embed-websocket")
    }
    implementation(spring.spring.springBootStarterLog4j2)
    implementation(spring.spring.springBootStarterValidation)
    implementation(spring.spring.springBootStarterActuator)
    implementation(spring.spring.springBootStarterJdbc)
    implementation(spring.spring.springRetry)
    implementation(spring.postgresql.postgresql)
    implementation(spring.flywaydb.flywayDatabasePostgresql) {
        exclude("com.fasterxml.jackson.dataformat")
        exclude("com.fasterxml.jackson.datatype")
    }
    implementation(spring.projectreactor.reactorNettyHttp)
    implementation(spring.jackson.jacksonModuleKotlin)
    implementation(spring.jackson.jacksonModuleBlackbird)
    implementation(spring.micrometer.micrometerRegistryOtlp)
    implementation(coroutines.kotlinx.kotlinxCoroutinesCoreJvm)
    implementation(coroutines.kotlinx.kotlinxCoroutinesSlf4j)
    implementation(kot.kotlin.kotlinReflect)
    implementation(libs.kotlin.logging)
    implementation(libs.disruptor)
    implementation(libs.protobuf.java)
    implementation(libs.otel.proto)
    compileOnly(libs.wiremock)

    testImplementation(spring.spring.springBootStarterTest)
    testImplementation(kot.kotlin.kotlinTestJunit5)
}

configurations.all {

    exclude(group = "org.springframework.boot", module = "spring-boot-starter-logging")
    resolutionStrategy {
        cacheDynamicVersionsFor(0, "minutes")
        cacheChangingModulesFor(0, "minutes")
    }
}

tasks.withType<KotlinCompile>().configureEach {
    compilerOptions {
        javaParameters = true
        freeCompilerArgs.addAll("-Xjsr305=strict", "-Xemit-jvm-type-annotations", "-Xannotation-default-target=param-property", "-Xjvm-default=all")
    }
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

tasks.bootJar {
    layered {
        enabled = true
    }
}

tasks.withType<JibTask> {
    notCompatibleWithConfigurationCache("Jib does not support the Gradle configuration cache yet")
}

//tasks.matching { it.name == "jib" || it.name == "jibDockerBuild" }.configureEach {
//    dependsOn(tasks.named("processAot"))
//}

/*
jmh {
    warmupIterations = 2
    iterations = 5
    fork = 1
    benchmarkMode = listOf("thrpt")
    timeUnit = "s"
}
*/

allOpen {
    annotation("org.springframework.context.annotation.Lazy")
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
