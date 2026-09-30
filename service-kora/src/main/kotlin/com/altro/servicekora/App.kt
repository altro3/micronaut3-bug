package com.altro.servicekora

import com.altro.servicekora.model.UserEntity
import io.koraframework.application.graph.KoraApplication
import io.koraframework.cache.caffeine.CaffeineCacheModule
import io.koraframework.common.annotation.KoraApp
import io.koraframework.config.hocon.HoconConfigModule
import io.koraframework.database.flyway.FlywayJdbcDatabaseModule
import io.koraframework.database.jdbc.JdbcDatabaseModule
import io.koraframework.http.server.undertow.UndertowPublicHttpServerModule
import io.koraframework.json.common.JsonModule
import io.koraframework.logging.common.LoggingModule
import io.koraframework.logging.common.masking.MaskingRules
import io.koraframework.logging.common.masking.MaskingStrategy
import io.koraframework.logging.logback.LogbackModule
import io.koraframework.openapi.management.OpenApiManagementModule
import io.koraframework.resilient.ResilientModule
import io.koraframework.validation.module.ValidationModule

@KoraApp
interface App :
    HoconConfigModule,
    JsonModule,
    LoggingModule,
    LogbackModule,
    UndertowPublicHttpServerModule,
    JdbcDatabaseModule,
    FlywayJdbcDatabaseModule,
    CaffeineCacheModule,
    OpenApiManagementModule,
    ResilientModule,
    ValidationModule {

    fun longMaskingRules(): MaskingRules<Long> {
        return MaskingRules(Long::class.java, emptyMap())
    }

    fun stringMaskingRules(): MaskingRules<String> {
        return MaskingRules(String::class.java, emptyMap())
    }

    @Suppress("UNCHECKED_CAST")
    fun userEntityMaskingRules2(): MaskingRules<UserEntity?> {
        return userEntityMaskingRules1() as MaskingRules<UserEntity?>
    }

    fun userEntityMaskingRules1(): MaskingRules<UserEntity> {
        return MaskingRules(UserEntity::class.java, emptyMap<String, MaskingStrategy>())
    }
}

fun main() {
    KoraApplication.run(AppGraph::graph)
}
