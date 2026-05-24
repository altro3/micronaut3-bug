plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

dependencies {
    implementation(libs.kotlin.logging)
    implementation(spring.spring.springBootAutoconfigure)
    implementation(spring.spring.springBootFlyway)
    implementation(spring.spring.springJdbc)
    implementation(spring.flywaydb.flywayDatabasePostgresql)
}
