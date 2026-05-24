plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
}

dependencies {
    implementation(spring.spring.springWebmvc)
    implementation(projects.mytracelog)
    implementation(projects.myutil)
}
