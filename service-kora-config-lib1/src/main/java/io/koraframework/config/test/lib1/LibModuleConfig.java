package io.koraframework.config.test.lib1;

import io.koraframework.config.common.annotation.ConfigMapper;
import io.koraframework.config.common.annotation.ConfigSource;
import java.time.Duration;

@ConfigSource("libModule")
public interface LibModuleConfig {
    boolean enabled();
    Duration timeout();
    RetryConfig retry();

    @ConfigMapper
    interface RetryConfig {
        int maxAttempts();
        String delay();
    }
}
