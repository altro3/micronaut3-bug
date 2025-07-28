plugins {
    id("org.jetbrains.kotlin.jvm") version "2.2.0"
//    id("org.jetbrains.kotlin.kapt") version "2.2.0"
    id("com.google.devtools.ksp") version "2.2.0-2.0.2"
//    id("org.jetbrains.kotlin.jvm") version "1.9.25"
//    id("org.jetbrains.kotlin.kapt") version "1.9.25"
//    id("com.google.devtools.ksp") version "1.9.25-1.0.20"
    id("io.micronaut.application") version "4.5.4"
    id("io.micronaut.openapi") version "4.5.4"
//    id "io.micronaut.aot" version "4.5.4"
//    id "org.openapi.generator" version "7.14.0"
}

val ver = mapOf(
    "micronaut" to "4.9.1",
    "core" to "4.9.8",
    "openapi" to "6.17.3",
    "serde" to "2.15.0",
)

//mainClassName = "com.micronaut.bug.Application"
micronaut {
    version("4.9.1")
    runtime("netty")
    testRuntime("junit5")
    enableNativeImage(false)
    processing {
        group(project.group.toString())
        incremental(true)
        annotations("com.micronaut.bug.*")
    }
}

dependencies {

    ksp("io.micronaut.validation:micronaut-validation-processor")
    ksp("io.micronaut.serde:micronaut-serde-processor")
    ksp("io.micronaut:micronaut-inject-kotlin")
    ksp("io.micronaut.openapi:micronaut-openapi")
    ksp("io.micronaut.security:micronaut-security-annotations")

//    kapt("io.micronaut.validation:micronaut-validation-processor")
//    kapt("io.micronaut.serde:micronaut-serde-processor")
//    kapt("io.micronaut:micronaut-inject-java")
//    kapt("io.micronaut.openapi:micronaut-openapi:${ver["openapi"]}")
//    kapt("io.micronaut.security:micronaut-security-annotations")

    compileOnly("io.micronaut:micronaut-inject-java")
    compileOnly("io.micronaut:micronaut-inject-kotlin")
    compileOnly("io.micronaut.openapi:micronaut-openapi-annotations")
    compileOnly("com.fasterxml.jackson.core:jackson-annotations")

    implementation("io.micronaut.validation:micronaut-validation")
    implementation("io.micronaut:micronaut-http-server-netty")
    implementation("io.micronaut:micronaut-http-client")
    implementation("io.micronaut:micronaut-management")
    implementation("io.micronaut.data:micronaut-data-model")
    implementation("io.micronaut:micronaut-runtime")
    implementation("io.micronaut.reactor:micronaut-reactor")
    implementation("io.micronaut.serde:micronaut-serde-jackson")
    implementation("org.jetbrains.kotlin:kotlin-stdlib")
    implementation("org.jetbrains.kotlin:kotlin-reflect")
    implementation("io.micronaut.security:micronaut-security-jwt")
    implementation("io.micronaut.security:micronaut-security-oauth2")

//    runtimeOnly("io.micronaut:micronaut-jackson-databind")
    runtimeOnly("io.micronaut.serde:micronaut-serde-jackson")
    runtimeOnly("ch.qos.logback:logback-classic")
    runtimeOnly("org.yaml:snakeyaml")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
}

kotlin {
    jvmToolchain(17)
}

ksp {
    arg("micronaut.openapi.project.dir", projectDir.toString())
}

tasks.jar {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}


configurations.configureEach {
    resolutionStrategy {
//        dependencySubstitution {
//            substitute(module("io.micronaut:micronaut-jackson-databind"))
//                    .using(module("io.micronaut.serde:micronaut-serde-jackson:1.3.2"))
//        }

        cacheDynamicVersionsFor(0, "minutes")
        cacheChangingModulesFor(0, "minutes")
    }
}
