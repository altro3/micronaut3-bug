package com.micronaut.bug.config.trace.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.net.URI
import java.time.Duration

@ConfigurationProperties(prefix = "app.trace.tempo")
data class TraceProperties(
    /**
     * Включить/выключить экспорт в Tempo.
     */
    val enabled: Boolean = false,
    /**
     * URL эндпоинта Tempo (OTLP HTTP Protobuf).
     */
    val url: URI = URI.create("http://localhost:4318/v1/traces"),
    /**
     * Таймаут на установку соединения.
     */
    val connectTimeout: Duration = Duration.ofSeconds(2),
    /**
     * Максимальное количество спанов в одном батче.
     * Оптимально: 512. Большие значения экономят трафик, но увеличивают потребление памяти.
     */
    val batchSize: Int = 512,
    /**
     * Интервал принудительной отправки батча, даже если он не заполнился до [batchSize].
     * Определяет задержку появления трейсов в интерфейсе Grafana.
     */
    val flushInterval: Duration = Duration.ofSeconds(2),
    /**
     * Максимальный размер внутренней очереди спанов.
     * При переполнении новые спаны будут отбрасываться для защиты приложения от OutOfMemory.
     */
    val queueCapacity: Int = 10000,
    /**
     * Время ожидания завершения отправки оставшихся в очереди спанов при выключении приложения.
     */
    val shutdownTimeout: Duration = Duration.ofSeconds(5),
    /**
     * Пауза перед следующей попыткой после возникновения ошибки в цикле экспорта.
     * Предотвращает "спам" в логи и излишнюю нагрузку при недоступности Tempo.
     */
    val retryInterval: Duration = Duration.ofSeconds(1),
)
