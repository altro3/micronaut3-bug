package com.micronaut.bug.trace.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.NestedConfigurationProperty
import java.net.URI
import java.time.Duration

@ConfigurationProperties("app.trace")
class TraceProperties(
    /**
     * Глобальный флаг включения трейсинга в приложении.
     * Если false — NanoTracer не будет генерировать спаны и MDC.
     */
    val enabled: Boolean = true,
    /**
     * Порог "медленного" запроса.
     * Если выполнение длится дольше — трейс будет отправлен в Tempo принудительно,
     * игнорируя настройки сэмплирования (sampleRate), и получит метку 'slow_request'.
     */
    val slowRequestThreshold: Duration = Duration.ofSeconds(10),
    /**
     * Вероятность экспорта успешных трейсов в Tempo.
     * Значение от 0.0 (ничего не слать) до 1.0 (слать всё).
     * Ошибочные трейсы всегда игнорируют этот параметр и шлются на 100%.
     */
    val sampleRate: Double = 1.0,
    /**
     * Настройки пакетного экспортера в Grafana Tempo.
     */
    @NestedConfigurationProperty
    val exporter: ExporterProperties = ExporterProperties(),
    /**
     * Заголовки запроса, которые наследуются всеми дочерними спанами
     * и пробрасываются в исходящие HTTP-вызовы (Metadata/Baggage).
     */
    val propagationHeaders: Set<String> = setOf(),
) {

    class ExporterProperties(
        /**
         * Включить/выключить отправку данных в Tempo.
         */
        val enabled: Boolean = true,
        /**
         * URL эндпоинта Tempo (OTLP HTTP Protobuf).
         */
        val url: URI = URI.create("http://localhost:4318/v1/traces"),
        /**
         * Таймаут на установку соединения.
         */
        val connectTimeout: Duration = Duration.ofSeconds(2),
        /**
         * Таймаут на запрос.
         */
        val requestTimeout: Duration = Duration.ofSeconds(2),
        /**
         * Максимальное количество спанов в одном батче.
         */
        val batchSize: Int = 512,
        /**
         * Интервал принудительной отправки батча.
         */
        val flushInterval: Duration = Duration.ofSeconds(2),
        /**
         * Максимальный размер внутренней очереди (защита от OOM).
         */
        val queueCapacity: Int = 10000,
        /**
         * Время на очистку очереди при выключении (Graceful Shutdown).
         */
        val shutdownTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Пауза при ошибках в цикле экспорта (Backoff).
         */
        val retryInterval: Duration = Duration.ofSeconds(1)
    )
}
