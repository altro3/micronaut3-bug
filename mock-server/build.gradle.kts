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
    implementation(kot.kotlin.kotlinReflect)
    implementation(libs.wiremock)
    implementation(projects.mycommon)
    implementation(projects.myutil)
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
