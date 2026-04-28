package com.micronaut.bug.config.log

import jakarta.validation.Valid
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.core.Ordered
import org.springframework.util.unit.DataSize
import java.net.URI
import java.time.Duration

/**
 * Настройки логирования входящих (контроллеры) и исходящих (клиенты) HTTP-запросов.
 * Позволяют гибко управлять детализацией, безопасностью памяти и форматом вывода.
 */
@ConfigurationProperties("app.log")
class LogProperties(
    /**
     * Включает или выключает логирование на уровне контроллеров (ServerLoggingFilter).
     */
    var enabled: Boolean = true,
    /**
     * Порядок выполнения фильтра в цепочке Spring Security / Servlet Filter.
     * По умолчанию имеет наивысший приоритет.
     */
    var order: Int = Ordered.HIGHEST_PRECEDENCE + 1,
    /**
     * Если true, запросы к Spring Boot Actuator (/actuator) не будут логироваться.
     */
    var skipActuator: Boolean = true,
    /**
     * Максимальный размер тела, который мы готовы прочитать в память.
     * Если тело больше — кэширование не производится, логируется заглушка.
     */
    @field:Positive
    var maxPayloadSize: DataSize = DataSize.ofMegabytes(15),
    /**
     * Максимально допустимый размер тела сообщения в логах (в байтах).
     * Значения 0 или -1 отключают ограничение (логируется весь объем).
     */
    @field:Min(-1)
    var limitLogSize: Int = 8192,
    /**
     * Количество байт, которые будут взяты с начала и с конца сообщения
     * при выполнении обрезки (truncate) длинного лога.
     */
    @field:Positive
    var truncateChunkSize: Int = 4096,
    /**
     * Включает форматирование вывода (Pretty Print для JSON, разбор Multipart).
     * Если false — данные выводятся как есть (сырая строка).
     */
    @field:NotNull
    var prettyPrint: Boolean = true,
    /**
     * Список ключей из MDC (Mapped Diagnostic Context), значения которых будут
     * дополнительно выведены в текстовый блок лога.
     * Полезно для визуальной связки лога с конкретным контекстом (например, userId, traceId).
     */
    var mdcKeys: List<String> = emptyList(),
    /**
     * Настройки интеграции с Grafana Loki.
     */
    @field:Valid
    var loki: LokiProperties = LokiProperties()
) {

    class LokiProperties(
        /**
         * Включает отправку логов напрямую в Loki через loki-logback-appender.
         */
        var enabled: Boolean = true,
        /**
         * URL эндпоинта Loki для пуша логов (Protobuf/HTTP).
         */
        var url: URI = URI.create("http://localhost:3100/loki/api/v1/push"),
        /**
         * Включает расширенное логирование внутренней работы аппендера.
         */
        var verbose: Boolean = false,
        /**
         * Включает использование бинарного Protobuf API для отправки логов.
         */
        var useProtobufApi: Boolean = true,
        /**
         * Паттерн для формирования структурированных метаданных (Structured Metadata).
         * Позволяет передавать высококардинальные данные (traceId, userId) без раздувания индекса Loki.
         * Пример: "traceId=%mdc{rqId}\spanId=%mdc{extRqId}\ntargetId=%mdc{targetId}\nclient=%mdc{client}\ntarget=%mdc{target}"
         */
        var structuredMetadata: String? = "traceId=%mdc{rqId}\nspanId=%mdc{extRqId}\ntargetId=%mdc{targetId}\nclient=%mdc{client}\ntarget=%mdc{target}",
        /**
         * Добавляет специальные маркеры чтения в поток логов.
         */
        var readMarkers: Boolean = false,
        /**
         * Включает сбор и экспорт внутренних метрик производительности.
         */
        var metricsEnabled: Boolean = false,
        /**
         * Размер батча (количество логов) перед отправкой.
         */
        @field:Positive
        var batchSize: Int = 200,
        /**
         * Максимальный объем памяти для накопления батча перед отправкой.
         */
        var batchMaxBytes: DataSize = DataSize.ofMegabytes(4),
        /**
         * Интервал времени, по истечении которого неполный батч будет принудительно отправлен.
         */
        var batchTimeout: Duration = Duration.ofSeconds(60),
        /**
         * Максимальный размер очереди отправки в байтах.
         */
        var sendQueueMaxBytes: DataSize = DataSize.ofMegabytes(40),
        /**
         * Количество попыток повторной отправки при сетевых сбоях.
         */
        @field:Positive
        var maxRetries: Int = 2,
        /**
         * Минимальная задержка перед повторной попыткой отправки.
         */
        var minRetryBackoff: Duration = Duration.ofMillis(500),
        /**
         * Максимальная задержка перед повторной попыткой отправки.
         */
        var maxRetryBackoff: Duration = Duration.ofSeconds(60),
        /**
         * Случайное отклонение для времени повторной попытки.
         */
        var maxRetryJitter: Duration = Duration.ofMillis(500),
        /**
         * Таймаут на установку соединения с сервером.
         */
        var connectionTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Таймаут на выполнение HTTP-запроса на пуш логов.
         */
        var requestTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Время бездействия потока отправки, после которого он будет завершен.
         */
        var threadExpirationTimeout: Duration = Duration.ofMinutes(5),
        /**
         * Если true, пакеты будут отбрасываться при получении ошибки 429.
         */
        var dropRateLimitedBatches: Boolean = false,
        /**
         * Интервал проверки состояния внутренних очередей.
         */
        var internalQueuesCheckTimeout: Duration = Duration.ofMillis(25),
        /**
         * Использовать ли Direct Buffers для снижения нагрузки на GC.
         */
        var useDirectBuffers: Boolean = true,
        /**
         * Отправить ли остатки логов из очереди при выключении приложения.
         */
        var drainOnStop: Boolean = true,
        /**
         * Флаг использования статических меток для оптимизации.
         */
        var staticLabels: Boolean = true,
    )
}
