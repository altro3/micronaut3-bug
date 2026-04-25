package com.micronaut.bug.config.log

import com.micronaut.bug.config.log.MdcConverter.Companion.TARGET_ID
import com.micronaut.bug.config.log.MdcConverter.Companion.X_REQ_ID
import jakarta.validation.Valid
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
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
    var enabledControllerLogging: Boolean = true,
    /**
     * Порядок выполнения фильтра в цепочке Spring Security / Servlet Filter.
     * По умолчанию имеет наивысший приоритет.
     */
    var order: Int = Ordered.HIGHEST_PRECEDENCE,
    /**
     * Если true, запросы к Spring Boot Actuator (/actuator) не будут логироваться.
     */
    var skipActuator: Boolean = true,
    /**
     * Максимальный размер тела, который мы готовы прочитать в память.
     * Если тело больше — кэширование не производится, логируется заглушка.
     */
    @field:Positive
    var maxPayloadSize: DataSize = DataSize.ofMegabytes(15), // 15MB по умолчанию
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
        var enabled: Boolean = false,
        /**
         * URL эндпоинта Loki для пуша логов (Protobuf/HTTP).
         */
        var url: URI = URI.create("http://localhost"),
        /**
         * Имя приложения для метки 'app' в Loki.
         */
        @field:NotBlank
        var appName: String = "my-service",
        /**
         * Размер батча (количество логов) перед отправкой.
         */
        @field:Positive
        var batchSize: Int = 200,
        /**
         * Максимальный объем памяти (DataSize) для накопления батча перед отправкой.
         * Дополнительная защита от OOM при тяжелых логах.
         */
        var batchMaxBytes: DataSize = DataSize.ofMegabytes(4),
        /**
         * Интервал времени, по истечении которого неполный батч будет принудительно отправлен.
         */
        var batchTimeout: Duration = Duration.ofSeconds(60),
        /**
         * Флаг использования статических меток. Оптимизирует производительность, если набор меток не меняется.
         */
        var staticLabels: Boolean = false,
        /**
         * Максимальный размер оперативной памяти, выделяемый под очередь отправки при недоступности сервера.
         */
        var sendQueueMaxBytes: DataSize = DataSize.ofMegabytes(40),
        /**
         * Максимальное количество попыток повторной отправки при сетевых сбоях.
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
         * Случайное отклонение (jitter) для времени повторной попытки.
         */
        var maxRetryJitter: Duration = Duration.ofMillis(500),
        /**
         * Если true, пакеты будут отбрасываться при получении ошибки 429 (Rate Limited) от Loki.
         */
        var dropRateLimitedBatches: Boolean = false,
        /**
         * Таймаут проверки состояния внутренних очередей приложения.
         */
        var internalQueuesCheckTimeout: Duration = Duration.ofMillis(25),
        /**
         * Включает использование Direct Buffers для снижения нагрузки на Garbage Collector.
         */
        var useDirectBuffers: Boolean = true,
        /**
         * Если true, приложение попытается отправить все оставшиеся в очереди логи перед завершением процесса.
         */
        var drainOnStop: Boolean = true,
        /**
         * Включает экспорт внутренних метрик Loki4j для мониторинга производительности аппендера.
         */
        var metricsEnabled: Boolean = false,
        /**
         * Список ключей из MDC, которые станут индексируемыми метками (labels) в Loki.
         * Пример: ["x-req-id", "targetId"]
         */
        var labels: List<String> = listOf(X_REQ_ID, TARGET_ID),
        /**
         * Время ожидания установки соединения с сервером Loki.
         */
        var connectionTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Максимальное время ожидания ответа от Loki на запрос отправки логов.
         */
        var requestTimeout: Duration = Duration.ofSeconds(5),
        /**
         * Время бездействия потока отправки, после которого он будет завершен.
         * Используется в JavaHttpConfig для управления ресурсами.
         */
        var threadExpirationTimeout: Duration = Duration.ofMinutes(5),
    )
}
