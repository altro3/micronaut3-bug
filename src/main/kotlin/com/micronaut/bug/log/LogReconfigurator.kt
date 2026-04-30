package com.micronaut.bug.log

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.ConsoleAppender
import ch.qos.logback.core.CoreConstants
import ch.qos.logback.core.OutputStreamAppender
import com.fasterxml.jackson.databind.ObjectMapper
import com.github.loki4j.logback.JavaHttpSender
import com.github.loki4j.logback.JsonLayout
import com.github.loki4j.logback.Loki4jAppender
import com.github.loki4j.logback.PipelineConfigAppenderBase
import com.micronaut.bug.log.config.LogProperties
import com.micronaut.bug.log.otlp.OtlpAppender
import com.micronaut.bug.log.otlp.OtlpEncoder
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.core.env.Environment
import org.springframework.util.ClassUtils

/**
 * Реконфигуратор Logback, выполняющий динамическую настройку консольного вывода
 * и регистрацию внешних систем сбора логов (Loki, OTLP).
 */
class LogReconfigurator(
    private val environment: Environment,
    private val props: LogProperties,
    mapper: ObjectMapper
) {

    private val logMasker = LogMasker(props, mapper)

    /**
     * Выполняет переконфигурацию после полной готовности приложения.
     */
    @EventListener(ApplicationReadyEvent::class)
    fun reconfigureLogback() {
        val loggerContext = LoggerFactory.getILoggerFactory() as? LoggerContext ?: return

        val appName = environment.getProperty(PROP_APP_NAME) ?: VAL_UNKNOWN_APP
        val nodeName = environment.getProperty(PROP_NODE_NAME) ?: VAL_UNKNOWN_NODE

        // 1. Настройка MDC и консольного паттерна
        configureConsole(loggerContext)

        // 2. Добавляем OTLP (VictoriaLogs), если включен
        if (props.otlp.enabled) {
            setupOtlpAppender(loggerContext, appName, nodeName)
        }

        // 3. Добавляем Loki только если он включен И библиотека присутствует в classpath
        if (props.loki.enabled && isLokiPresent()) {
            setupLokiAppender(loggerContext, appName, nodeName)
        }
    }

    /**
     * Проверяет наличие библиотеки loki4j в classpath.
     */
    private fun isLokiPresent(): Boolean = ClassUtils.isPresent(
        "com.github.loki4j.logback.Loki4jAppender",
        this.javaClass.classLoader
    )

    /**
     * Обновляет паттерн вывода для всех ConsoleAppender и регистрирует mdcBlock.
     */
    private fun configureConsole(loggerContext: LoggerContext) {
        MdcConverter.keys = props.mdcKeys

        @Suppress("UNCHECKED_CAST")
        val rules = loggerContext.getObject(CoreConstants.PATTERN_RULE_REGISTRY) as? MutableMap<String, String>
            ?: mutableMapOf<String, String>().also { loggerContext.putObject(CoreConstants.PATTERN_RULE_REGISTRY, it) }

        rules[MDC_BLOCK_WORD] = MdcConverter::class.java.name

        val pattern = environment.getProperty(CONSOLE_PATTERN_KEY) ?: DEFAULT_PATTERN

        loggerContext.loggerList.forEach { logger ->
            val appenderIterator = logger.iteratorForAppenders()
            while (appenderIterator.hasNext()) {
                val appender = appenderIterator.next()
                if (appender is ConsoleAppender<*>) {
                    val streamAppender = appender as OutputStreamAppender<ILoggingEvent>
                    val encoder = streamAppender.encoder as? PatternLayoutEncoder
                    encoder?.let {
                        it.pattern = pattern
                        it.start()
                    }
                }
            }
        }
    }

    /**
     * Программная инициализация Loki.
     */
    private fun setupLokiAppender(loggerContext: LoggerContext, appName: String, nodeName: String) {
        val rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME)
        if (rootLogger.getAppender(APPENDER_NAME_LOKI) != null) {
            return
        }

        val lokiProps = props.loki
        val appender = Loki4jAppender().apply {
            name = APPENDER_NAME_LOKI
            context = loggerContext
            setLabels("app=$appName\nnode=$nodeName")
            setMessage(JsonLayout().apply {
                context = loggerContext
                start()
            })
            setHttp(PipelineConfigAppenderBase.HttpCfg().apply {
                setUrl(lokiProps.url.toString())
                setConnectionTimeoutMs(lokiProps.connectionTimeout.toMillis())
                setRequestTimeoutMs(lokiProps.requestTimeout.toMillis())
                setSender(JavaHttpSender().apply {
                    setInnerThreadsExpirationMs(lokiProps.threadExpirationTimeout.toMillis())
                })
            })
            setBatch(PipelineConfigAppenderBase.BatchCfg().apply {
                setMaxItems(lokiProps.batchSize)
                setMaxBytes(lokiProps.batchMaxBytes.toBytes().toInt())
                setTimeoutMs(lokiProps.batchTimeout.toMillis())
            })
            start()
        }
        rootLogger.addAppender(appender)
    }

    /**
     * Инициализация кастомного OTLP аппендера.
     */
    private fun setupOtlpAppender(loggerContext: LoggerContext, appName: String, nodeName: String) {
        val rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME)
        if (rootLogger.getAppender(APPENDER_NAME_OTLP) != null) {
            return
        }

        // Создаем энкодер здесь, прокидывая в него все зависимости
        val otlpEncoder = OtlpEncoder(appName, nodeName, logMasker)

        // Передаем готовый энкодер в аппендер
        val appender = OtlpAppender(props, otlpEncoder).apply {
            name = APPENDER_NAME_OTLP
            context = loggerContext
            start()
        }
        rootLogger.addAppender(appender)
    }

    companion object {
        private const val PROP_APP_NAME = "spring.application.name"
        private const val PROP_NODE_NAME = "app.node.name"
        private const val CONSOLE_PATTERN_KEY = "app.log.pattern.console"

        private const val VAL_UNKNOWN_APP = "unknown-app"
        private const val VAL_UNKNOWN_NODE = "unknown-node"

        private const val MDC_BLOCK_WORD = "mdcBlock"
        private const val DEFAULT_PATTERN = "%d{HH:mm:ss.SSS} %highlight(%-5level) %magenta(%$MDC_BLOCK_WORD) [%thread] %cyan(%logger{25}) - %msg%n%throwable"

        private const val APPENDER_NAME_LOKI = "LOKI"
        private const val APPENDER_NAME_OTLP = "OTLP"
    }
}
