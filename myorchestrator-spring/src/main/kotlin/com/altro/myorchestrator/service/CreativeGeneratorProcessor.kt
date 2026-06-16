package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Creative
import com.altro.myorchestrator.api.dto.Platform
import io.github.oshai.kotlinlogging.KotlinLogging
import org.springframework.stereotype.Component
import tools.jackson.databind.json.JsonMapper
import tools.jackson.module.kotlin.jacksonTypeRef

@Component
class CreativeGeneratorProcessor(
    private val aiInferenceService: AiInferenceService,
    private val jsonMapper: JsonMapper,
) {

    private val log = KotlinLogging.logger {}

    /**
     * Генерирует один рекламный текст (Creative) под конкретную выбранную платформу
     */
    fun generate(platform: Platform, briefText: String, moderationRules: String): Creative {
        log.info { "Запуск ИИ-генерации текста креатива для платформы: \${platform.name}" }

        val systemInstruction = """
            Вы — изолированный API-компонент генерации текстов. Вам ЗАПРЕЩЕНО общаться с пользователем, писать вступления или заключения.
            Вы должны вернуть ответ СТРОГО в формате валидного JSON-объекта, соответствующего структуре.
            НЕ используйте markdown-разметку и кавычки ```json. Начните ответ сразу со знака {.
            
            ВАЖНО: Пиши рекламный текст максимально емко и коротко. Общий объем ответа должен строго укладываться в 150 токенов. Сразу после закрытия структуры JSON прекращай генерацию (вызови <|im_end|>).
            
            Структура JSON:
            {
              "title": "сгенерированный заголовок",
              "bodyText": "сгенерированный текст объявления"
            }
        """.trimIndent()

        val userPrompt = """
            Рекламная платформа: ${platform.name}
            На основе следующего брифа/описания продукта: '$briefText',
            напиши короткий рекламный текст (заголовок title и текст объявления bodyText).
            
            ОБЯЗАТЕЛЬНО ПРИМЕНИ ПРАВИЛА МОДЕРАЦИИ ИЗ БАЗЫ ЗНАНИЙ (лимиты символов):
            $moderationRules
        """.trimIndent()

        // Вызов Qwen-35B с нашим намертво вшитым n_predict лимитом токенов в кастомных атрибутах
        val finalRawJson = aiInferenceService.generateCreativesJson(systemInstruction, userPrompt)

        return try {
            // Десериализуем плоскую мапу прямо в твой базовый класс Creative
            val rawMap = jsonMapper.readValue(
                finalRawJson,
                jacksonTypeRef<Map<String, String>>()
            )

            Creative(
                title = rawMap["title"] ?: "Специальное предложение",
                bodyText = rawMap["bodyText"] ?: "Узнайте подробности на нашем сайте."
            )
        } catch (e: Exception) {
            log.error(e) { "Ошибка десериализации креатива ИИ. Ответ от модели был:\n$finalRawJson" }
            throw IllegalStateException("Модель выдала невалидную JSON структуру рекламного объявления на Шаге 3", e)
        }
    }
}
