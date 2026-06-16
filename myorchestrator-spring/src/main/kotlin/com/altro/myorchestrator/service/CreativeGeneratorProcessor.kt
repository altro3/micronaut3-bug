package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.Platform
import com.altro.myorchestrator.model.CampaignSession.CreativeWithImage
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

    fun generate(analysisResult: AnalysisResult, moderationRules: String): Map<Platform, CreativeWithImage> {
        val platformsList = analysisResult.recommendations.map { it.platform }.joinToString()

        val systemInstruction = """
    Вы — изолированный API-компонент генерации текстов. Вам ЗАПРЕЩЕНО общаться с пользователем.
    Вы должны вернуть ответ СТРОГО в формате валидного JSON-объекта.
    НЕ используйте markdown-разметку и кавычки ```json. Начните ответ сразу со знака {.
    
    ВАЖНО: Пиши рекламный текст максимально емко и коротко. Общий объем ответа должен строго укладываться в 300 токенов. Сразу после закрытия структуры JSON прекращай генерацию (вызови <|im_end|>).
    
    Структура JSON:
    {
      "VK_ADS": { "title": "текст", "bodyText": "текст" },
      "YANDEX_DIRECT": { "title": "текст", "bodyText": "текст" }
    }
""".trimIndent()

        val userPrompt = """
            На основе описания аудитории: '${analysisResult.audienceDescription}', напиши тексты объявлений для платформ: $platformsList.
            Обязательно учти правила модерации:
            $moderationRules
        """.trimIndent()

        val finalRawJson = aiInferenceService.generateCreativesJson(systemInstruction, userPrompt)

        return try {
            val rawMap = jsonMapper.readValue(
                finalRawJson,
                jacksonTypeRef<Map<Platform, Map<String, String>>>()
            )
            rawMap.mapValues { (_, value) ->
                CreativeWithImage(
                    title = value["title"] ?: "",
                    bodyText = value["bodyText"] ?: "",
                    imageUrl = "http://localhost:8083/api/v1/images/default-stub.jpg"
                )
            }
        } catch (e: Exception) {
            log.error(e) { "Ошибка десериализации креативов ИИ. Ответ: $finalRawJson" }
            throw IllegalStateException("Модель выдала невалидную JSON структуру на Шаге 3", e)
        }
    }
}
