package com.micronaut.bug.config;

import io.netty.handler.codec.http.HttpContentDecompressor;
import org.springframework.boot.web.embedded.netty.NettyServerCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class NettyDecompressionConfig {

    @Bean
    public NettyServerCustomizer decompressingCustomizer() {
        return server -> server.doOnConnection(connection ->
                // Добавляем стандартный декомпрессор Netty в начало цепочки обработки
                connection.addHandlerFirst("decompressor", new HttpContentDecompressor(10000))
        );
    }
}