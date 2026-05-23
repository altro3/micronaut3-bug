plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
}

dependencies {
    implementation(spring.slf4j.slf4jApi)
    implementation(spring.jakarta.jakartaValidationApi)
    implementation(spring.spring.springWeb)
    implementation(spring.spring.springBootStarterJson)
    implementation(spring.jackson.jacksonModuleKotlin)
    implementation(spring.jackson.jacksonModuleBlackbird)
    implementation(project(":mytracelog"))
}
