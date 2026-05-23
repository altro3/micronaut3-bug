package com.altro.common.log

import com.altro.common.log.LogReconfigurator.OtlpInitializer.setup
import com.altro.common.log.config.LogProperties
import com.altro.common.log.otlp.OtlpAppender
import com.altro.common.log.otlp.OtlpLogEncoder
import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.core.LoggerContext
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.core.env.Environment
import org.springframework.util.ClassUtils

class LogReconfigurator(
    private val environment: Environment,
    private val props: LogProperties,
) {

    private val logMasker = LogMasker(props)

    @EventListener(ApplicationReadyEvent::class)
    fun reconfigure() {
        val ctx = LogManager.getContext(false) as? LoggerContext ?: return

        val appName = environment.getProperty(PROP_APP_NAME) ?: VAL_UNKNOWN_APP
        val nodeName = environment.getProperty(PROP_NODE_NAME) ?: VAL_UNKNOWN_NODE

        // 1. Настройка MDC (передаем ключи в наш Garbage-Free конвертер)
        MdcConverter.keys = props.mdcKeys

        // 2. Настройка сокращения стектрейса для консоли
        SmartExConverter.maxLines = props.stackTraceMaxLines
        SmartExConverter.rootCauseFull = props.stackTraceRootCauseFull

        // 2. Добавляем OTLP (VictoriaLogs), если он включен в конфигурации И библиотека присутствует в classpath
        if (props.otlp.enabled && isOtlpPresent()) {
            setup(ctx, appName, nodeName, props, logMasker)
        }

        // Применяем изменения ко всем активным логгерам
        ctx.updateLoggers()
    }

    private fun isOtlpPresent(): Boolean = ClassUtils.isPresent(
        "io.opentelemetry.proto.logs.v1.LogRecord",
        this.javaClass.classLoader
    )

    private object OtlpInitializer {

        fun setup(ctx: LoggerContext, appName: String, nodeName: String, props: LogProperties, logMasker: LogMasker) {
            val config = ctx.configuration

            // Если аппендер уже есть — не плодим дубликаты
            if (config.appenders.containsKey(APPENDER_NAME_OTLP)) return

            // Создаем энкодер (адаптированный под LogEvent)
            val otlpLogEncoder = OtlpLogEncoder(
                appName = appName,
                nodeName = nodeName,
                props = props,
                logMasker = logMasker,
            )

            val appender = OtlpAppender(
                name = APPENDER_NAME_OTLP,
                props = props,
                encoder = otlpLogEncoder,
            )

            appender.start()
            config.addAppender(appender)

            // Добавляем аппендер в Root логгер
            val rootConfig = config.getRootLogger()
            rootConfig.addAppender(appender, null, null)
        }
    }

    companion object {
        private const val PROP_APP_NAME = "spring.application.name"
        private const val PROP_NODE_NAME = "app.node.name"

        private const val VAL_UNKNOWN_APP = "unknown-app"
        private const val VAL_UNKNOWN_NODE = "unknown-node"

        private const val APPENDER_NAME_OTLP = "OTLP"
    }
}
