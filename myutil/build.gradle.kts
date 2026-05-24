plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.spring)
}

dependencies {

    api(spring.jackson3.jacksonDatabind)
    api(spring.jackson3.jacksonModuleKotlin)
    api(spring.jackson3.jacksonModuleBlackbird)
}
