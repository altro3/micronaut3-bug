plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

dependencies {
    api(spring.spring.springBootFlyway)

    implementation(spring.spring.springBootAutoconfigure)
    implementation(spring.spring.springJdbc)
    implementation(libs.kotlin.logging)
}
