plugins {
    `java-library`
    `maven-publish`
}

group = "io.koraframework.config.test"
version = "1.0.0-SNAPSHOT"

dependencies {
    annotationProcessor(kora.koraframework.annotationProcessors)

    implementation(kora.koraframework.configCommon)
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            // Имя артефакта берем равным названию твоего модуля
            artifactId = "service-kora-config-lib1"
        }
    }
}
