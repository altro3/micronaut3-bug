import dev.aga.gradle.versioncatalogs.Generator.generate

plugins {
    id("dev.aga.gradle.version-catalog-generator") version "4.2.2"
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven { url = uri("https://s01.oss.sonatype.org/content/repositories/snapshots/") }
        maven { url = uri("https://central.sonatype.com/repository/maven-snapshots/") }
        mavenLocal()
    }

    versionCatalogs {
        generate("spring") { fromToml("spring-boot-dependencies") }
        generate("kt") { fromToml("kotlin") }
        generate("coroutines") { fromToml("coroutines") }
        generate("springAi") { fromToml("spring-ai") }
        generate("kora") { fromToml("kora") }
    }
}

rootProject.name = "micronaut3-bug"

include(
    "mycommon",
    "myflyway",
    "myhttp",
    "mytracelog",
    "myutil",
    "mock-server",
    "myorchestrator-spring",
    "mymcp-server",
    "service-kora",
    "service-vk",
    "service-yad",
)

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
