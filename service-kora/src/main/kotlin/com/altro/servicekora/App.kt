package com.altro.servicekora

import io.koraframework.application.graph.KoraApplication
import io.koraframework.cache.caffeine.CaffeineCacheModule
import io.koraframework.common.annotation.KoraApp
import io.koraframework.config.hocon.HoconConfigModule
import io.koraframework.database.jdbc.JdbcDatabaseModule
import io.koraframework.http.server.undertow.UndertowPublicHttpServerModule
import io.koraframework.json.common.JsonModule
import io.koraframework.logging.logback.LogbackModule
import io.koraframework.resilient.ResilientModule

@KoraApp
interface App :
    HoconConfigModule,
    JsonModule,
    LogbackModule,
    UndertowPublicHttpServerModule,
    JdbcDatabaseModule,
    CaffeineCacheModule,
    ResilientModule

fun main() {
    KoraApplication.run(AppGraph::graph)
}
