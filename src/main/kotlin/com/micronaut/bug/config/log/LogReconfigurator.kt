package com.micronaut.bug.config.log

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.ConsoleAppender
import ch.qos.logback.core.CoreConstants
import ch.qos.logback.core.OutputStreamAppender
import ch.qos.logback.core.util.StatusPrinter2
import com.github.loki4j.logback.JavaHttpSender
import com.github.loki4j.logback.JsonLayout
import com.github.loki4j.logback.Loki4jAppender
import com.github.loki4j.logback.PipelineConfigAppenderBase
import org.slf4j.LoggerFactory
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.core.env.Environment

/**
 * Реконфигуратор логбэка, выполняющий настройку консольного вывода и
 * динамическую регистрацию аппендера Grafana Loki.
 */
class LogReconfigurator(
    private val environment: Environment,
    private val props: LogProperties
) {

    /**
     * Основная точка входа для переконфигурации после полной готовности приложения.
     * Выполняет настройку кастомных конвертеров, обновление паттернов консоли
     * и инициализацию Loki-аппендера.
     */
    @EventListener(ApplicationReadyEvent::class)
    fun reconfigureLogback() {
        val loggerContext = LoggerFactory.getILoggerFactory() as? LoggerContext ?: return

        // 1. Передаем список ключей из конфигурации в конвертер
        MdcConverter.keys = props.mdcKeys

        // 2. Регистрируем правило конвертации для паттерна
        @Suppress("UNCHECKED_CAST")
        val rules = loggerContext.getObject(CoreConstants.PATTERN_RULE_REGISTRY) as? MutableMap<String, String>
            ?: mutableMapOf<String, String>().also { loggerContext.putObject(CoreConstants.PATTERN_RULE_REGISTRY, it) }
        rules[MDC_BLOCK_WORD] = MdcConverter::class.java.name

        // 3. Получаем паттерн из свойств или используем дефолтный
        val pattern = environment.getProperty(CONSOLE_PATTERN_KEY) ?: DEFAULT_PATTERN

        // 4. Перезапускаем все ConsoleAppender-ы с новым паттерном
        loggerContext.loggerList.forEach { logger ->
            val appenderIterator = logger.iteratorForAppenders()
            while (appenderIterator.hasNext()) {
                val appender = appenderIterator.next()
                if (appender is ConsoleAppender<*>) {
                    val streamAppender = appender as OutputStreamAppender<ILoggingEvent>
                    val encoder = streamAppender.encoder as? PatternLayoutEncoder
                    encoder?.let {
                        it.pattern = pattern
                        it.start() // Перезапуск применяет новое правило mdcBlock
                    }
                }
            }
        }

        // 3. Добавляем Loki, если включен
        if (props.loki.enabled) {
            setupLokiAppender(loggerContext)
        }
    }

    /**
     * Программная инициализация Loki4jAppender.
     * Настраивает отправку логов в бинарном формате Protobuf с учетом
     * ограничений по размеру батча для предотвращения OOM при тяжелых логах.
     *
     * @param loggerContext Контекст исполнения Logback.
     */
    private fun setupLokiAppender(loggerContext: LoggerContext) {
        val lokiProps = props.loki
        val appName = environment.getProperty("spring.application.name") ?: "unknown-app"
        val nodeName = environment.getProperty("app.node.name") ?: "unknown-node"
        val rootLogger = loggerContext.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)

        // Проверяем, не добавлен ли уже аппендер (защита от повторной инициализации)
        if (rootLogger.getAppender(LOKI_APPENDER_NAME) != null) {
            return
        }

        val appender = Loki4jAppender().apply {
            name = LOKI_APPENDER_NAME
            context = loggerContext
            setLabels("app=$appName\nnode=$nodeName")
            setMessage(JsonLayout().apply {
                context = loggerContext
                start()
            })
            setReadMarkers(lokiProps.readMarkers)
            setVerbose(lokiProps.verbose)
            setStructuredMetadata(lokiProps.structuredMetadata)
            setMetricsEnabled(lokiProps.metricsEnabled)
            setHttp(PipelineConfigAppenderBase.HttpCfg().apply {
                setUrl(lokiProps.url.toString())
                setConnectionTimeoutMs(lokiProps.connectionTimeout.toMillis())
                setRequestTimeoutMs(lokiProps.requestTimeout.toMillis())
                setMaxRetries(lokiProps.maxRetries)
                setMinRetryBackoffMs(lokiProps.minRetryBackoff.toMillis())
                setMaxRetryBackoffMs(lokiProps.maxRetryBackoff.toMillis())
                setMaxRetryJitterMs(lokiProps.maxRetryJitter.toMillis().toInt())
                setDropRateLimitedBatches(lokiProps.dropRateLimitedBatches)
                setUseProtobufApi(lokiProps.useProtobufApi)
                setSender(JavaHttpSender().apply {
                    setInnerThreadsExpirationMs(lokiProps.threadExpirationTimeout.toMillis())
                })
            })
            setBatch(PipelineConfigAppenderBase.BatchCfg().apply {
                setMaxItems(lokiProps.batchSize)
                setMaxBytes(lokiProps.batchMaxBytes.toBytes().toInt())
                setTimeoutMs(lokiProps.batchTimeout.toMillis())
                setSendQueueMaxBytes(lokiProps.sendQueueMaxBytes.toBytes())
                setInternalQueuesCheckTimeoutMs(lokiProps.internalQueuesCheckTimeout.toMillis())
                setDrainOnStop(lokiProps.drainOnStop)
                setUseDirectBuffers(lokiProps.useDirectBuffers)
                setStaticLabels(lokiProps.staticLabels)
            })
            start()
        }

        rootLogger.addAppender(appender)
    }

    companion object {
        // Слово, которое будет использоваться в паттерне: %mdcBlock
        private const val MDC_BLOCK_WORD = "mdcBlock"

        // Ключ в application.yml для настройки паттерна
        private const val CONSOLE_PATTERN_KEY = "app.log.pattern.console"

        // Дефолтный паттерн: Дата Уровень [MDC] [Поток] Логгер - Сообщение
        private const val DEFAULT_PATTERN = "%d{HH:mm:ss.SSS} %highlight(%-5level) %magenta(%$MDC_BLOCK_WORD) [%thread] %cyan(%logger{25}) - %msg%n%throwable"

        private const val LOKI_APPENDER_NAME = "LOKI"
    }
}
