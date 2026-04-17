package com.micronaut.bug.config.log

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.ConsoleAppender
import ch.qos.logback.core.CoreConstants
import ch.qos.logback.core.OutputStreamAppender
import org.slf4j.LoggerFactory
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.core.env.Environment

class LogReconfigurator(
    private val environment: Environment,
    private val props: LogProperties
) {

    @EventListener(ApplicationReadyEvent::class)
    fun reconfigureLogback() {
        val loggerContext = LoggerFactory.getILoggerFactory() as? LoggerContext ?: return

        // 1. Передаем список ключей из конфигурации в конвертер
        MdcConverter.keys = props.mdcKeys

        // 2. Регистрируем правило конвертации для паттерна
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
    }

    companion object {
        // Слово, которое будет использоваться в паттерне: %mdcBlock
        private const val MDC_BLOCK_WORD = "mdcBlock"

        // Ключ в application.yml для настройки паттерна
        private const val CONSOLE_PATTERN_KEY = "app.log.pattern.console"

        // Дефолтный паттерн: Дата Уровень [MDC] [Поток] Логгер - Сообщение
        private const val DEFAULT_PATTERN =
            "%d{HH:mm:ss.SSS} %highlight(%-5level) %magenta(%$MDC_BLOCK_WORD) [%thread] %cyan(%logger{25}) - %msg%n%throwable"
    }
}
