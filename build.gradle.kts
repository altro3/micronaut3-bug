plugins {
    id("io.micronaut.application") version "4.6.0"
//    id("io.micronaut.openapi") version "4.6.0"
//    id("org.jetbrains.kotlin.jvm") version "2.2.10"
//    id("org.jetbrains.kotlin.kapt") version "2.2.10"
//    id("com.google.devtools.ksp") version "2.2.10-2.0.2"
    id("org.jetbrains.kotlin.jvm") version "1.9.25"
    id("org.jetbrains.kotlin.kapt") version "1.9.25"
    id("com.google.devtools.ksp") version "1.9.25-1.0.20"
//    id "io.micronaut.aot" version "4.5.4"
//    id "org.openapi.generator" version "7.14.0"
}

val ver = mapOf(
    "micronaut" to "4.10.0",
    "core" to "4.10.7",
    "openapi" to "6.19.1",
    "serde" to "2.15.0",
)

//mainClassName = "com.micronaut.bug.Application"
micronaut {
    version(ver["micronaut"])
    runtime("netty")
    testRuntime("junit5")
    enableNativeImage(false)
    processing {
        group(project.group.toString())
        incremental(false)
        annotations("com.micronaut.bug.*")
    }
//    openapi {
//        version = ver["openapi"]
//        server(file("swagger.yml")) {
//
//        }
//    }
}

dependencies {

//    ksp("io.micronaut.validation:micronaut-validation-processor")
////    ksp("io.micronaut.serde:micronaut-serde-processor")
//    ksp("io.micronaut:micronaut-inject-kotlin")
//    ksp("io.micronaut.openapi:micronaut-openapi")
//    ksp("io.micronaut.security:micronaut-security-annotations")

    kapt("io.micronaut.validation:micronaut-validation-processor")
//    kapt("io.micronaut.serde:micronaut-serde-processor")
    kapt("io.micronaut:micronaut-inject-java")
    kapt("io.micronaut.openapi:micronaut-openapi:${ver["openapi"]}")
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
    implementation("io.micronaut:micronaut-jackson-databind")
//    implementation("io.micronaut.serde:micronaut-serde-jackson")
    implementation("org.jetbrains.kotlin:kotlin-stdlib")
    implementation("org.jetbrains.kotlin:kotlin-reflect")
//    implementation("io.micronaut.security:micronaut-security-jwt")
//    implementation("io.micronaut.security:micronaut-security-oauth2")

    runtimeOnly("ch.qos.logback:logback-classic")
    runtimeOnly("org.yaml:snakeyaml")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
}

kotlin {
    jvmToolchain(17)
}

//ksp {
//    arg("micronaut.openapi.project.dir", projectDir.toString())
//}

kapt {
    arguments {
        arg("micronaut.openapi.project.dir", "$projectDir")
    }
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

tasks.register("removeMnFiles") {
    doLast {
        delete(layout.buildDirectory.dir("tmp/kapt3/classes/main/META-INF/micronaut"))
        delete(
            layout.buildDirectory.dir("tmp/kapt3/classes/main").get().asFileTree.matching {
                include("**/*\$Definition*.class", "**/*\$Introspection.class")
            }.files
        )
    }
    dependsOn(tasks.named("kaptKotlin"))
}
tasks.compileKotlin {
    dependsOn(tasks.named("removeMnFiles"))
}
