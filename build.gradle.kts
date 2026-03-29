plugins {
    id("io.micronaut.application") version "4.6.2"
//    id("io.micronaut.openapi") version "4.6.2"
//    id "io.micronaut.aot" version "4.6.2"
//    id("org.jetbrains.kotlin.jvm") version "2.2.10"
//    id("org.jetbrains.kotlin.kapt") version "2.2.10"
//    id("com.google.devtools.ksp") version "2.2.10-2.0.2"
    id("org.jetbrains.kotlin.jvm") version "1.9.25"
    id("org.jetbrains.kotlin.kapt") version "1.9.25"
//    id("com.google.devtools.ksp") version "1.9.25-1.0.20"
//    id "org.openapi.generator" version "7.19.0"
}

val ver = mapOf(
    "kotlin" to "1.9.25",
    "micronaut" to "4.10.10",
    "core" to "4.10.18",
    "openapi" to "6.20.0",
    "serde" to "2.16.2",
    "jackson" to "2.21.2",
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
//            lang = "kotlin"
//        }
//    }
}

dependencies {

//    ksp("io.micronaut.validation:micronaut-validation-processor")
////    ksp("io.micronaut.serde:micronaut-serde-processor")
//    ksp("io.micronaut:micronaut-inject-kotlin")
//    ksp("io.micronaut.openapi:micronaut-openapi")
//    ksp("io.micronaut.security:micronaut-security-annotations")

    kapt(platform("io.micronaut:micronaut-core-bom:${ver["core"]}"))
    kapt("io.micronaut.validation:micronaut-validation-processor:${ver["core"]}")
//    kapt("io.micronaut.serde:micronaut-serde-processor")
    kapt("io.micronaut:micronaut-inject-java:${ver["core"]}")
    kapt("io.micronaut.openapi:micronaut-openapi:${ver["openapi"]}")
//    kapt("io.micronaut.security:micronaut-security-annotations")

    compileOnly("io.micronaut.openapi:micronaut-openapi:${ver["openapi"]}")
//    compileOnly("jakarta.annotation:jakarta.annotation-api")
    compileOnly("io.micronaut:micronaut-inject-java")
    compileOnly("io.micronaut:micronaut-inject-kotlin")
    compileOnly("io.micronaut.openapi:micronaut-openapi-annotations:${ver["openapi"]}")
    compileOnly("com.fasterxml.jackson.core:jackson-databind:${ver["jackson"]}")

    implementation(platform("io.micronaut:micronaut-core-bom:${ver["core"]}"))
    implementation("io.micronaut.validation:micronaut-validation")
    implementation("io.micronaut:micronaut-http-server-netty")
    implementation("io.micronaut:micronaut-http-client")
    implementation("io.micronaut:micronaut-management")
    implementation("io.micronaut:micronaut-runtime")
    implementation("io.micronaut.reactor:micronaut-reactor")
    implementation("io.micronaut:micronaut-jackson-databind")
    implementation("io.micronaut.serde:micronaut-serde-jackson")
    implementation("org.jetbrains.kotlin:kotlin-stdlib:${ver["kotlin"]}")
    implementation("org.jetbrains.kotlin:kotlin-reflect:${ver["kotlin"]}")
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
