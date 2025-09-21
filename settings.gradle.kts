dependencyResolutionManagement {
    repositories {
        mavenCentral()
        mavenLocal()
        mavenCentral {
            url = uri("https://s01.oss.sonatype.org/content/repositories/snapshots/")
        }
    }
}

pluginManagement {
    repositories {
        mavenLocal()
        gradlePluginPortal()
    }
}

rootProject.name = "micronaut3-bug"
