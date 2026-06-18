package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service

@Service
class PlatformRouter(
    @Qualifier("routerAgentClient") private val routerAgentClient: ChatClient
) {
    private val log = KotlinLogging.logger {}

    fun determineTargetPlatform(session: CampaignSession, userInput: String): Platform {
        var currentPlatform = session.platform
        val isUserForcingSwitch = isUserAskingForPlatformSwitch(userInput)

        log.info { "🧩 [Router] Входной стейт: $currentPlatform, принудительный триггер: $isUserForcingSwitch" }

        if (currentPlatform == null || (currentPlatform == Platform.CHAT && isUserForcingSwitch)) {
            log.info { "🧩 [Router] Запуск ИИ-классификатора намерений..." }
            val newPlatform = routeUserInputToPlatform(userInput)

            if (newPlatform != currentPlatform) {
                log.info { "🔄 [Router] МГНОВЕННАЯ ДЕЛЕГАЦИЯ: Смена агента на $newPlatform" }
                session.platform = newPlatform
                if (newPlatform != Platform.CHAT) {
                    session.campaignId = null
                }
                return newPlatform
            }
        }
        return currentPlatform ?: Platform.CHAT
    }

    private fun routeUserInputToPlatform(userInput: String): Platform {
        return try {
            val routerPrompt = """
                Проанализируй сообщение пользователя и классифицируй его намерение.
                Выдай в ответ строго одно слово из трех вариантов:
                1. YANDEX_DIRECT — если пользователь явно просит настроить, запустить или создать рекламу в Яндекс.Директ.
                2. VK_ADS — если пользователь явно просит настроить, запустить или создать рекламу во ВКонтакте (VK Ads).
                3. CHAT — если пользователь просто здоровается, задает общие вопросы, просит совета или просто общается.
                
                Сообщение пользователя: '$userInput'
                Твой ответ (строго одно слово из трех вариантов):
            """.trimIndent()

            val response = routerAgentClient.prompt()
                .user(routerPrompt)
                .call()
                .content()
                ?.trim()?.uppercase() ?: "CHAT"

            when {
                response.contains("YANDEX") -> Platform.YANDEX_DIRECT
                response.contains("VK") -> Platform.VK_ADS
                else -> Platform.CHAT
            }
        } catch (e: Exception) {
            log.error(e) { "Ошибка ИИ-маршрутизации трафика, фолбек на Platform.CHAT" }
            Platform.CHAT
        }
    }

    private fun isUserAskingForPlatformSwitch(text: String): Boolean {
        val lower = text.lowercase()
        return lower.contains("яндекс") || lower.contains("директ") || lower.contains("вк ") || lower.contains("vk") || lower.contains("вконтакте") || lower.contains("янде")
    }
}
