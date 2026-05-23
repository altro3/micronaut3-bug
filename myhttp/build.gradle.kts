plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
}

dependencies {
    implementation(spring.slf4j.slf4jApi)
    implementation(spring.jakarta.jakartaValidationApi)
    implementation(spring.spring.springBootStarterWeb)
    implementation(spring.spring.springRetry)
    implementation(spring.projectreactor.reactorNettyHttp)
    implementation(spring.jackson.jacksonModuleKotlin)
    implementation(spring.jackson.jacksonModuleBlackbird)
    implementation(project(":mytracelog"))
}
