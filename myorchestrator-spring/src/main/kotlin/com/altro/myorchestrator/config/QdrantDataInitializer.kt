package com.altro.myorchestrator.config

import com.altro.myorchestrator.model.Platform
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
            Document(
                UUID.randomUUID().toString(),
                "Правила YANDEX_DIRECT: Заголовок объявления до 56 символов. Текст объявления до 81 символа. Запрещен CapsLock. Текст должен быть строго на русском языке, без синтаксических ошибок.",
                mapOf("platform" to Platform.YANDEX_DIRECT.name)
            ),

            Document(
                UUID.randomUUID().toString(),
                "Правила VK_ADS: Заголовок объявления от 3 до 30 символов. Основной текст до 90 символов. Запрещены эмодзи в заголовках. Обращение к пользователю строго на 'Вы'.",
                mapOf("platform" to Platform.VK_ADS.name)
            ),

            Document(
                UUID.randomUUID().toString(),
                "Закон о маркировке интернет-рекламы в РФ: Любое объявление обязано содержать пометку 'Реклама' и указание ИНН рекламодателя (например: 'Реклама. ИНН 1234567890').",
                mapOf("platform" to "ALL")
            )
        )

        try {
            vectorStore.accept(documents)
        } catch (e: Exception) {
            println("Ошибка инициализации Qdrant (возможно, коллекция уже создана): ${e.message}")
        }
    }
}
