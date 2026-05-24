plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
}

dependencies {
    implementation(spring.slf4j.slf4jApi)
    implementation(spring.jakarta.jakartaValidationApi)
    implementation(spring.spring.springBootStarterWeb)
    implementation(spring.projectreactor.reactorNettyHttp)
    implementation(projects.mytracelog)
    implementation(projects.myutil)
}
