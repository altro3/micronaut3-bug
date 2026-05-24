plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

dependencies {
    api(spring.spring.springBootFlyway)
    api(libs.flyway.core)
    api(libs.flyway.database.postgresql)

    implementation(spring.spring.springBootAutoconfigure)
    implementation(spring.spring.springJdbc)
    implementation(libs.kotlin.logging)
}
