package io.koraframework.benchmark.java;

import io.koraframework.application.graph.*;
import io.koraframework.common.annotation.*;
//import io.koraframework.config.hocon.*;
//import io.koraframework.http.server.undertow.*;
import io.koraframework.json.common.*;
import io.koraframework.logging.common.*;
import io.koraframework.logging.logback.*;
import io.koraframework.validation.module.*;
import io.koraframework.config.aot.generated.AotModule;
import io.koraframework.http.server.netty.NettyPublicHttpServerModule;



@KoraApp
public interface App extends
    AotModule,
    NettyPublicHttpServerModule,
//    HoconConfigModule,
//    UndertowPublicHttpServerModule,
    JsonModule,
    LoggingModule,
    LogbackModule,
    ValidationModule {
    static void main(String[] args) {
        KoraApplication.run(AppGraph::graph);
    }
}