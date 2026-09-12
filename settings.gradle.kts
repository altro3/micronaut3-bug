import dev.aga.gradle.versioncatalogs.Generator.generate

plugins {
    id("dev.aga.gradle.version-catalog-generator") version "4.2.2"
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven {
            url = uri("https://s01.oss.sonatype.org/content/repositories/snapshots/")
        }
        mavenLocal()
    }

    versionCatalogs {
        generate("spring") { fromToml("spring-boot-dependencies") }
        generate("kt") { fromToml("kotlin") }
        generate("coroutines") { fromToml("coroutines") }
        generate("springAi") { fromToml("spring-ai") }
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
    "service-vk",
    "service-yad",
)

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
