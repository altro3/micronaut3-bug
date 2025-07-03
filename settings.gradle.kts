dependencyResolutionManagement {
    repositories {
        mavenCentral()
        mavenLocal()
        mavenCentral {
            url = uri("https://s01.oss.sonatype.org/content/repositories/snapshots/")
        }
    }
}

rootProject.name = "micronaut3-bug"
