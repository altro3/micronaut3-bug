package com.micronaut.bug.log

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.ConsoleAppender
import ch.qos.logback.core.CoreConstants
import ch.qos.logback.core.OutputStreamAppender
import com.github.loki4j.logback.JavaHttpSender
import com.github.loki4j.logback.JsonLayout
import com.github.loki4j.logback.Loki4jAppender
import com.github.loki4j.logback.PipelineConfigAppenderBase
import com.micronaut.bug.log.LogReconfigurator.Companion.MDC_BLOCK_WORD
import com.micronaut.bug.log.LogReconfigurator.LokiInitializer.setup
import com.micronaut.bug.log.LogReconfigurator.OtlpInitializer.setup
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
 * и регистрацию внешних систем сбора логов (Loki, OTLP) в рантайме.
 *
 * Позволяет отказаться от статических XML-файлов конфигурации логирования,
 * обеспечивая гибкое управление логированием через стандартный механизм свойств Spring Boot.
 *
 * Особенности реализации:
 * - Безопасен к отсутствию библиотек логирования (Loki, OTLP) в Classpath.
 * - Применяет паттерны форматирования на лету для всех активных консольных аппендеров.
 * - Инкапсулирует инициализацию внешних систем во вложенные объекты для ленивой загрузки классов (Lazy Loading).
 *
 * @property environment Окружение Spring для извлечения системных свойств и переменных окружения.
 * @property props Свойства логирования, загруженные из конфигурации приложения.
 */
class LogReconfigurator(
    private val environment: Environment,
    private val props: LogProperties,
) {

    /**
     * Экземпляр маскера, используемый для очистки логов от чувствительных данных
     * перед отправкой во внешние системы агрегации.
     */
    private val logMasker = LogMasker(props)

    /**
     * Выполняет переконфигурацию системы логирования после того, как контекст приложения
     * полностью поднялся и готов к обработке трафика.
     *
     * Метод гарантирует, что все динамические свойства (например, имя хоста или динамический порт)
     * уже разрешены и доступны для использования в метаданных логов.
     */
    @EventListener(ApplicationReadyEvent::class)
    fun reconfigureLogback() {
        val loggerContext = LoggerFactory.getILoggerFactory() as? LoggerContext ?: return

        val appName = environment.getProperty(PROP_APP_NAME) ?: VAL_UNKNOWN_APP
        val nodeName = environment.getProperty(PROP_NODE_NAME) ?: VAL_UNKNOWN_NODE

        // 1. Настройка MDC и консольного паттерна
        configureConsole(loggerContext)

        // 2. Добавляем OTLP (VictoriaLogs), если он включен в конфигурации И библиотека присутствует в classpath
        if (props.otlp.enabled && isOtlpPresent()) {
            setup(loggerContext, appName, nodeName, props, logMasker)
        }

        // 3. Добавляем Loki, если он включен в конфигурации И библиотека присутствует в classpath
        if (props.loki.enabled && isLokiPresent()) {
            setup(loggerContext, appName, nodeName, props)
        }
    }

    /**
     * Проверяет наличие библиотеки `loki4j` в classpath приложения.
     *
     * @return `true`, если основной класс аппендера Loki доступен для загрузки.
     */
    private fun isLokiPresent(): Boolean = ClassUtils.isPresent(
        "com.github.loki4j.logback.Loki4jAppender",
        this.javaClass.classLoader
    )

    /**
     * Проверяет наличие OTLP Protobuf моделей в classpath приложения.
     *
     * @return `true`, если основной класс Protobuf-модели LogRecord доступен для загрузки.
     */
    private fun isOtlpPresent(): Boolean = ClassUtils.isPresent(
        "io.opentelemetry.proto.logs.v1.LogRecord",
        this.javaClass.classLoader
    )

    /**
     * Динамически обновляет паттерн вывода для всех обнаруженных [ConsoleAppender]
     * и регистрирует кастомное ключевое слово [MDC_BLOCK_WORD] в реестре правил Logback.
     *
     * Это позволяет выводить в консоль красиво отформатированный блок MDC-параметров
     * согласно правилам, описанным в [MdcConverter].
     *
     * @param loggerContext Контекст Logback для регистрации правил и поиска аппендеров.
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
     * Вспомогательный объект-инициализатор для Grafana Loki.
     *
     * Вынесен в `object` для обеспечения ленивой загрузки классов (Lazy Loading).
     * JVM не будет пытаться разрешить внешние типы `Loki4jAppender` до тех пор,
     * пока не будет вызван метод [setup]. Это предотвращает падение приложения
     * с ошибкой [NoClassDefFoundError] при отсутствии библиотеки в classpath.
     */
    private object LokiInitializer {
        /**
         * Выполняет программную конфигурацию и старт аппендера Loki.
         *
         * @param loggerContext Текущий контекст логирования.
         * @param appName Имя приложения для проставления статических меток.
         * @param nodeName Имя ноды/хоста для проставления статических меток.
         * @param props Конфигурационные свойства с параметрами батчинга и сетевыми таймаутами.
         */
        fun setup(loggerContext: LoggerContext, appName: String, nodeName: String, props: LogProperties) {
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
    }

    /**
     * Вспомогательный объект-инициализатор для OpenTelemetry (OTLP).
     *
     * Вынесен в `object` для обеспечения ленивой загрузки классов (Lazy Loading).
     * JVM не будет пытаться разрешить внешние типы Protobuf-моделей до тех пор,
     * пока не будет вызван метод [setup]. Это предотвращает падение приложения
     * с ошибкой [NoClassDefFoundError] при отсутствии библиотеки в classpath.
     */
    private object OtlpInitializer {
        /**
         * Выполняет программную конфигурацию и старт аппендера OTLP.
         *
         * @param loggerContext Текущий контекст логирования.
         * @param appName Имя приложения для формирования метаданных ресурса.
         * @param nodeName Имя ноды/хоста для формирования метаданных ресурса.
         * @param props Конфигурационные свойства с параметрами батчинга и сетевыми таймаутами.
         * @param logMasker Маскировщик для очистки тела логов перед упаковкой в OTLP.
         */
        fun setup(loggerContext: LoggerContext, appName: String, nodeName: String, props: LogProperties, logMasker: LogMasker) {
            val rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME)
            if (rootLogger.getAppender(APPENDER_NAME_OTLP) != null) {
                return
            }

            val otlpEncoder = OtlpEncoder(appName, nodeName, logMasker)

            val appender = OtlpAppender(props, otlpEncoder).apply {
                name = APPENDER_NAME_OTLP
                context = loggerContext
                start()
            }
            rootLogger.addAppender(appender)
        }
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
