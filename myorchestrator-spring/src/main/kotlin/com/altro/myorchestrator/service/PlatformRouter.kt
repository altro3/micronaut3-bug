package com.altro.myorchestrator.service

import com.altro.myorchestrator.config.AiConfig.RouterOutput
import com.altro.myorchestrator.model.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class PlatformRouter(
    @Qualifier("routerAgentClient")
    private val routerAgentClient: ChatClient
) {
    private val log = KotlinLogging.logger {}

    fun determineTargetPlatform(currentPlatform: Platform?, userInput: String): Platform? {
        val isUserForcingSwitch = isUserAskingForPlatformSwitch(userInput)

        log.info { "🧩 [Router] Входной стейт: $currentPlatform, принудительный триггер: $isUserForcingSwitch" }

        if (currentPlatform == null || isUserForcingSwitch) {
            log.info { "🧩 [Router] Запуск ИИ-классификатора намерений..." }
            val determined = routeUserInputToPlatform(userInput)

            if (determined != currentPlatform && determined != null) {
                log.info { "🔄 [Router] МГНОВЕННАЯ ДЕЛЕГАЦИЯ: Смена платформы на $determined" }
                return determined
            }
        }
        return currentPlatform
    }

    private fun routeUserInputToPlatform(userInput: String): Platform? =
        try {
            routerAgentClient.prompt()
                .user(userInput)
                .call()
                .entity(RouterOutput::class.java)
                ?.platform
        } catch (e: Exception) {
            log.error(e) { "Ошибка ИИ-маршрутизации трафика, фолбек на null" }
            null
        }

    private fun isUserAskingForPlatformSwitch(text: String): Boolean {
        return PLATFORM_SWITCH_REGEX.containsMatchIn(text)
    }

    companion object {
        private val PLATFORM_SWITCH_REGEX = Regex(
            "яндекс|директ|вк|vk|вконтакте",
            RegexOption.IGNORE_CASE
        )
    }
}
