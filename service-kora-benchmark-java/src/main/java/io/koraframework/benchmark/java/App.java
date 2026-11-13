package io.koraframework.benchmark.java;

import io.koraframework.application.graph.KoraApplication;
import io.koraframework.common.annotation.KoraApp;
import io.koraframework.config.aot.generated.*;
import io.koraframework.http.server.netty.*;
//import io.koraframework.config.hocon.*;
//import io.koraframework.http.server.undertow.*;
import io.koraframework.json.common.JsonModule;
import io.koraframework.logging.common.LoggingModule;
import io.koraframework.logging.logback.LogbackModule;
import io.koraframework.validation.module.ValidationModule;

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
    static void main(String[] args) throws Exception {
//        runBenchmarkFastJson2(AppGraph.graph());
        KoraApplication.run(AppGraph::graph);
    }
}