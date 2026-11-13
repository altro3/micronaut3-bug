package io.koraframework.benchmark.java;

import io.koraframework.application.graph.*;
import io.koraframework.common.annotation.*;
import io.koraframework.config.aot.generated.*;
import io.koraframework.database.jdbc.*;
import io.koraframework.http.server.netty.*;
import io.koraframework.json.common.*;
import io.koraframework.logging.common.*;
import io.koraframework.logging.logback.*;
import io.koraframework.validation.module.*;

//import io.koraframework.config.hocon.HoconConfigModule;
//import io.koraframework.http.server.undertow.UndertowPublicHttpServerModule;
//import io.koraframework.database.flyway.FlywayJdbcDatabaseModule;

@KoraApp
public interface App extends
    AotModule,
    NettyPublicHttpServerModule,
//    HoconConfigModule,
//    UndertowPublicHttpServerModule,
//    FlywayJdbcDatabaseModule,
    JsonModule,
    LoggingModule,
    LogbackModule,
    ValidationModule,
    JdbcDatabaseModule {
    static void main(String[] args) {
        KoraApplication.run(AppGraph::graph);
    }
}