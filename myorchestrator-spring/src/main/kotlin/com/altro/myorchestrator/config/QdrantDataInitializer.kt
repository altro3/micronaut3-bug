package com.altro.myorchestrator.config

import com.altro.myorchestrator.api.dto.Platform
import org.springframework.ai.document.Document
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.boot.CommandLineRunner
import org.springframework.stereotype.Component
import java.util.UUID

@Component
class QdrantDataInitializer(
    private val vectorStore: VectorStore
) : CommandLineRunner {

    override fun run(vararg args: String) {

        val documents = listOf(
            // Правило 1: Модерация VK
            Document(
                UUID.randomUUID().toString(),
                "Требования VK_ADS: Длина заголовка объявления должна быть строго от 3 до 30 символов. Основной текст объявления (bodyText) должен быть строго до 90 символов включая пробелы. Запрещено использовать CapsLock.",
                mapOf("platform" to Platform.VK_ADS.name, "type" to "limit")
            ),
            // Правило 2: Модерация Яндекса
            Document(
                UUID.randomUUID().toString(),
                "Требования YANDEX_DIRECT: Длина основного заголовка текстово-графического объявления — до 56 символов. Длина текста объявления (bodyText) — строго до 81 символа. Обязательно использовать восклицательный знак в конце призыва.",
                mapOf("platform" to Platform.YANDEX_DIRECT.name, "type" to "limit")
            ),
            // Правило 3: Общее для общепита
            Document(
                UUID.randomUUID().toString(),
                "Общие правила для категории Кафе и Рестораны: При рекламе локальных акций на комбо-завтраки или ланчи, текст должен содержать упоминание стоимости (например, 'от 200 рублей') или условий доставки.",
                mapOf("platform" to "ALL", "type" to "category")
            )
        )

        // accept() автоматически вызывает локальный эмбеддер qwen-35B (2048) 
        // и сохраняет векторы с payload-метаданными в контейнер Qdrant
        vectorStore.accept(documents)
    }
}
