plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
    alias(libs.plugins.kotlin.kapt)
}

dependencies {
    kapt(spring.log4j.log4jCore)

    api(libs.kotlin.logging)
    api(libs.otel.proto)

    compileOnly(spring.spring.springBootJdbc)

    implementation(spring.spring.springBootStarterWeb)
    implementation(spring.spring.springBootStarterLog4j2)
    implementation(spring.jakarta.jakartaValidationApi)
    implementation(coroutines.kotlinx.kotlinxCoroutinesCoreJvm)
    implementation(coroutines.kotlinx.kotlinxCoroutinesSlf4j)
    implementation(libs.disruptor)
    implementation(libs.protobuf.java)
    implementation(projects.myutil)
}
