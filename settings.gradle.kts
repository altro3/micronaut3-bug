import dev.aga.gradle.versioncatalogs.Generator.generate

plugins {
    id("dev.aga.gradle.version-catalog-generator") version "4.2.0"
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
        generate("kot") { fromToml("kotlin") }
        generate("coroutines") { fromToml("coroutines") }
    }
}

rootProject.name = "micronaut3-bug"

include(
    "mycommon",
    "myflyway",
    "myhttp",
    "mytracelog",
    "myutil",
    "service1",
    "service2",
)

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
