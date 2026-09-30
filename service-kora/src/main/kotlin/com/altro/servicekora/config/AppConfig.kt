package com.altro.servicekora.config

import io.koraframework.config.common.annotation.ConfigSource

@ConfigSource("app")
interface AppConfig {

    fun name(): String

    fun version(): String

    fun environment(): String
}
