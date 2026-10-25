plugins {
    `java-library`
    `maven-publish`
}

group = "io.koraframework.config.test"
version = "1.0.0-SNAPSHOT"

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            artifactId = "service-kora-config-lib2"
        }
    }
}
