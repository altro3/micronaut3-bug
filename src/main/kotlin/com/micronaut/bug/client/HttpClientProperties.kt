package com.micronaut.bug.client

import com.micronaut.bug.client.HttpClientProperties.ClientType.INTERNAL
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_CONNECT_TIMEOUT
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_MAX_ATTEMPTS
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_READ_TIMEOUT
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_DELAY
import com.micronaut.bug.client.HttpClientUtils.DEFAULT_RETRY_MULTIPLIER
import jakarta.validation.Valid
import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Positive
import jakarta.validation.constraints.PositiveOrZero
import org.springframework.boot.context.properties.NestedConfigurationProperty
import org.springframework.validation.annotation.Validated
import java.net.URI
import java.time.Duration

@Validated
class HttpClientProperties(
    /**
     * Базовый URL целевого сервиса.
     */
    @field:NotNull
    var url: URI = URI(""),
    /**
     * Имя сервиса. Используется для идентификации клиента в логах и мониторинге.
     */
    var serviceName: String? = null,
    /**
     * Тип клиента (INTERNAL или EXTERNAL). Определяет логику авторизации и безопасности.
     */
    @field:NotNull
    var type: ClientType = INTERNAL,
    /**
     * Ключ API для авторизации запросов в заголовках.
     */
    var apiKey: String? = null,
    /**
     * Включение поддержки GZIP.
     * Если true, клиент отправляет 'Accept-Encoding: gzip' и автоматически распаковывает ответы.
     */
    @field:NotNull
    var compress: Boolean = false,
    /**
     * Флаг включения передачи контекста трассировки (Trace ID) в заголовках.
     */
    @field:NotNull
    var tracing: Boolean = true,
    /**
     * Настройки логирования запросов и ответов.
     */
    @field:NotNull
    @field:Valid
    @NestedConfigurationProperty
    var log: LogProperties = LogProperties(),
    /**
     * Максимальное количество попыток выполнения запроса:
     * 0 - бесконечно, 1 - без повторов, >1 - конкретное число попыток.
     */
    @field:PositiveOrZero
    var maxAttempts: Int = DEFAULT_MAX_ATTEMPTS,
    /**
     * Настройки политики повторных попыток (интервалы, множители).
     */
    @field:NotNull
    @field:Valid
    @NestedConfigurationProperty
    var retryBackoff: RetryBackoffProperties = RetryBackoffProperties(),
    /**
     * Таймаут на чтение данных из установленного соединения.
     */
    @field:NotNull
    var readTimeout: Duration = DEFAULT_READ_TIMEOUT,
    /**
     * Таймаут на установку TCP-соединения.
     */
    @field:NotNull
    var connectTimeout: Duration = DEFAULT_CONNECT_TIMEOUT,
    /**
     * Конфигурация прокси-сервера. Если null, прокси не используется.
     */
    @field:Valid
    @NestedConfigurationProperty
    var proxy: ProxyProperties? = null,
) {

    class LogProperties(
        /**
         * Флаг включения логирования запросов и ответов.
         */
        @field:NotNull
        var enabled: Boolean = true,
        /**
         * Если true, в логи будет записываться полный URL с Query-параметрами.
         */
        @field:NotNull
        var fullUrl: Boolean = true,
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
    )

    class RetryBackoffProperties(
        /**
         * Начальная задержка перед первой повторной попыткой.
         */
        val delay: Duration = DEFAULT_RETRY_DELAY,
        /**
         * Максимальный предел задержки между попытками.
         */
        val maxDelay: Duration? = null,
        /**
         * Множитель для экспоненциального роста задержки.
         * При значении 1.0 интервал остается фиксированным.
         */
        @field:Min(1)
        val multiplier: Double = DEFAULT_RETRY_MULTIPLIER,
        /**
         * Включение случайного разброса (jitter) в интервалах между попытками.
         */
        val random: Boolean = false,
    )

    class ProxyProperties(
        /**
         * Флаг принудительной активации прокси.
         */
        @field:NotNull
        var enabled: Boolean = true,
        /**
         * Хост или IP-адрес прокси-сервера.
         */
        @field:NotBlank
        var host: String,
        /**
         * Порт прокси-сервера.
         */
        @field:Max(Short.MAX_VALUE.toLong())
        @field:Positive
        var port: Int,
        /**
         * Логин для авторизации на прокси.
         */
        var username: String?,
        /**
         * Пароль для авторизации на прокси.
         */
        var password: String?,
        /**
         * Таймаут на соединение именно с прокси-сервером.
         */
        @field:NotNull
        var connectTimeout: Duration = DEFAULT_CONNECT_TIMEOUT,
    )

    enum class ClientType {
        INTERNAL,
        EXTERNAL,
    }
}
