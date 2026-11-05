package io.koraframework.benchmark.java;

import io.koraframework.application.graph.KoraApplication;
import io.koraframework.common.annotation.KoraApp;
import io.koraframework.config.aot.generated.*;
//import io.koraframework.config.hocon.HoconConfigModule;
import io.koraframework.json.common.JsonModule;
import io.koraframework.logging.common.LoggingModule;
import io.koraframework.logging.logback.LogbackModule;
import io.koraframework.http.server.undertow.UndertowPublicHttpServerModule;
import io.koraframework.validation.module.ValidationModule;

import static io.koraframework.benchmark.java.JsonInMemoryBenchmark.runBenchmark;

@KoraApp
public interface App extends
    AotModule,
//    HoconConfigModule,
    JsonModule,
    LoggingModule,
    LogbackModule,
    ValidationModule,
    UndertowPublicHttpServerModule {
    static void main(String[] args) throws Exception {
        var graph = AppGraph.graph();
        runBenchmark(graph);
//        KoraApplication.run(AppGraph::graph);
    }
}