package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.OrchestrationState
import com.altro.myorchestrator.api.dto.chat.ChatRq
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.ai.chat.client.ChatClient
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.scheduler.Schedulers

@Service
class ChatOrchestrator(
    chatClientBuilder: ChatClient.Builder,
    private val marketingEngine: MarketingOrchestratorEngine
) {

    private val log = KotlinLogging.logger {}

    private val chatClient = chatClientBuilder.build()

    fun orchestrateChatStream(rq: ChatRq): Flux<String> {
        val userPrompt = rq.messages.lastOrNull()?.content ?: ""

        // ТРИГГЕР: Если пользователь просит создать кампанию/бриф — запускаем агентный движок
        if (userPrompt.contains("кампани", ignoreCase = true) || userPrompt.contains("реклам", ignoreCase = true)) {
            return Flux.create { sink ->
                try {
                    marketingEngine.executeOrchestration(userPrompt) { state ->
                        // Превращаем стейты в красивый Markdown-текст для фронтенда
                        val markdownChunk = convertStateToMarkdown(state)
                        if (markdownChunk.isNotEmpty()) {
                            sink.next(markdownChunk)
                        }
                    }
                    sink.complete()
                } catch (e: Exception) {
                    log.error(e) { "Error" }
                    sink.next("\n\n❌ **Ошибка оркестратора:** ${e.message}\n")
                    sink.complete()
                }
            }.subscribeOn(Schedulers.boundedElastic()) // Запускаем в пул потоков, чтобы не блокировать Netty/Tomcat
        }

        // Старый сквозной чат (fallback), если это обычный вопрос
        return fallbackStream(rq)
    }

    private fun convertStateToMarkdown(state: OrchestrationState): String {
        return when (state) {
            is OrchestrationState.Initial -> "🚀 **Запуск маркетингового оркестратора...**\n\n"
            is OrchestrationState.Analyzing -> "🔍 *Шаг 1: Анализирую бриф и извлекаю сущности (Qwen 27B)...*\n\n"
            is OrchestrationState.CreativeGeneration -> {
                """
            📂 *Шаг 2: Поиск локальных правил модерации в Qdrant...*
            🎯 *Выделенная аудитория:* `${state.audienceDescription}`
            📡 *Целевые платформы:* ${state.platforms.joinToString { "`$it`" }}
            🎨 *Шаг 3: Генерация креативов с учетом ограничений площадок...*
            
            """.trimIndent()
            }

            is OrchestrationState.BudgetAndValidation -> {
                val sb = StringBuilder("\n\n✅ **Креативы успешно сгенерированы!**\n\n")
                state.creatives.forEach { (platform, creative) ->
                    sb.append("### 📊 Платформа: `$platform`\n\n")
                    sb.append("📌 **Заголовок:** ${creative.title ?: "Без заголовка"}\n\n")
                    sb.append("📝 **Текст креатива:**\n${creative.bodyText ?: "Пусто"}\n\n")
                }
                sb.append("⚙️ *Шаг 4: Параллельная отправка кампаний в кабинеты (Loom Virtual Threads)...*\n\n")
                sb.toString()
            }

            is OrchestrationState.Uploading -> ""
            is OrchestrationState.Completed -> {
                val sb = StringBuilder("🎉 **Оркестрация успешно завершена!**\n\n")
                sb.append("### 📝 Отчет о выполнении (Имитация MCP):\n\n")

                state.logs.forEach { log ->
                    val prefix = if (log.contains("Ошибка", ignoreCase = true)) "❌" else "🔹"
                    val cleanLog = log.replace("♦", "").trim()
                    sb.append("$prefix $cleanLog\n\n")
                }
                sb.toString()
            }
        }
    }

    private fun fallbackStream(rq: ChatRq): Flux<String> {
        // Здесь ваш старый код сквозного стриминга chatClient.prompt()...stream().content()
        return chatClient.prompt().user(rq.messages.last().content).stream().content()
    }
}
