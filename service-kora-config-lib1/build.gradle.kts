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

java {
    withSourcesJar()
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("mavenJava") {
                artifactId = "service-kora-config-lib1"
               from(components["java"])
            }
        }
    }
}
