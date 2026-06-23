package com.altro.myorchestrator.config

import com.altro.myorchestrator.model.Platform
import io.qdrant.client.QdrantClient
import org.springframework.ai.document.Document
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.boot.CommandLineRunner
import org.springframework.stereotype.Component
import java.util.UUID

@Component
class QdrantDataInitializer(
    private val vectorStore: VectorStore,
    private val qdrantClient: QdrantClient,
) : CommandLineRunner {

    override fun run(vararg args: String) {

        /*
                try {
                    qdrantClient.createCollectionAsync(
                        "ad_rules_collection",
                        VectorParams.newBuilder()
                            .setSize(1536)
                            .setDistance(Collections.Distance.Cosine)
                            .build()
                    ).get()
                } catch (e: Exception) {
                    //
                }
        */

        val lawName = "Федеральный закон № 347-ФЗ о маркировке интернет-рекламы в РФ"

        val documents = listOf(
            Document(
                UUID.randomUUID().toString(),
                """
        Лимиты контента YANDEX_DIRECT:
        - Длина заголовка объявления: до 56 символов.
        - Длина основного текста объявления: до 81 символа.
        - Запрещено использование CapsLock и синтаксических ошибок.
        - Язык объявлений: строго русский.
        """.trimIndent(),
                mapOf("platform" to Platform.YANDEX_DIRECT.name)
            ),

            Document(
                UUID.randomUUID().toString(),
                """
        Требование Яндекса к категории 'Автосервисы и ремонт':
        Реклама автосервисов и ремонта автомобилей требует обязательного указания возрастной маркировки (0+, 6+, 12+, 16+ или 18+). Без возрастной метки модерация отклонит кампанию.
        """.trimIndent(),
                mapOf("platform" to Platform.YANDEX_DIRECT.name)
            ),

            Document(
                UUID.randomUUID().toString(),
                """
        $lawName:
        - Каждое рекламное объявление в РФ обязано содержать пометку 'Реклама' и реальный ИНН рекламодателя (например: 'Реклама. ИНН 5406123456').
        - Запрещено использовать вымышленные ИНН или заглушки (0000000000).
        """.trimIndent(),
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
