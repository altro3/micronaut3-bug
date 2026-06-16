package com.altro.myorchestrator.service

import com.altro.myorchestrator.api.dto.chat.ChatRq
import com.altro.myorchestrator.model.CampaignCreationStep
import com.altro.myorchestrator.model.CampaignSession
import com.altro.myorchestrator.model.CampaignSession.SessionContext
import com.altro.myorchestrator.repository.CampaignSessionRepository
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.FluxSink
import reactor.core.scheduler.Schedulers
import java.time.Instant

@Service
class ChatOrchestrator(
    private val sessionRepository: CampaignSessionRepository,
    private val aiInferenceService: AiInferenceService,
    private val vectorStoreService: VectorStoreService,
    private val mcpPublishingService: McpPublishingService,
    private val creativeGeneratorProcessor: CreativeGeneratorProcessor
) {

    /**
     * Главная точка входа. Тонкий диспетчер стрима.
     */
    fun orchestrateChatStream(rq: ChatRq): Flux<String> {
        return Flux.create { sink ->
            try {
                // Обработка пустой истории (Стартовое приветствие)
                if (rq.messages.isEmpty()) {
                    sendWelcomeMessage(sink)
                    return@create
                }

                val firstMessage = rq.messages.first().content.trim()
                val userInput = rq.messages.last().content.trim()

                // Точечный поиск сессии в БД по тексту первого сообщения
                val session = sessionRepository.findByFirstMessage(firstMessage)

                // Маршрутизация на основе текущего состояния State Machine
                if (session == null) {
                    initializeNewSession(firstMessage, userInput, sink)
                } else {
                    executeSessionStep(session, userInput, sink)
                }

            } catch (e: Exception) {
                sink.next("❌ Критический сбой пайплайна чата: ${e.message}")
                sink.error(e)
            }
        }.subscribeOn(Schedulers.boundedElastic())
    }

    // ==========================================================================================
    // ИЗОЛИРОВАННЫЕ МЕТОДЫ ОБРАБОТКИ ШАГОВ (STATE MACHINE)
    // ==========================================================================================

    /**
     * ШАГ 0: Приветствие, если история чата пуста
     */
    private fun sendWelcomeMessage(sink: FluxSink<String>) {
        sink.next("👋 Привет! Я локальный ИИ-ассистент AdBroker.\n")
        sink.next("Давайте настроим вашу маркетинговую кампанию. **Укажите регионы таргетинга** через запятую (например: *Москва, Санкт-Петербург*):")
        sink.complete()
    }

    /**
     * ШАГ 1: Инициализация сессии и сохранение регионов
     */
    private fun initializeNewSession(firstMessage: String, userInput: String, sink: FluxSink<String>) {
        sink.next("⚙️ Инициализирую черновик кампании...\n")
        val regions = userInput.split(",").map { it.trim() }

        // Ищем, есть ли уже драфт, чтобы обновить его, или создаем с нуля
        val existingDraft = sessionRepository.findByFirstMessage(firstMessage)
        val baseId = existingDraft?.campaignId ?: 0L

        val session = sessionRepository.save(
            CampaignSession(
                campaignId = baseId, // Переиспользуем ID драфта, если он есть
                currentStep = CampaignCreationStep.REGION_SELECTION,
                context = SessionContext(
                    rawBriefText = firstMessage,
                    selectedRegions = regions
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("✅ Регионы приняты: ${session.context.selectedRegions.joinToString()}.\n\n")
        sink.next("🤖 **[Шаг 1/4]**: Теперь отправьте мне подробный **текстовый бриф** или описание вашего продукта:")
        sink.complete()
    }

    /**
     * Диспетчер шагов для уже существующей сессии
     */
    private fun executeSessionStep(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        when (session.currentStep) {
            CampaignCreationStep.DRAFT -> initializeNewSession(session.context.rawBriefText ?: "", userInput, sink)
            CampaignCreationStep.REGION_SELECTION -> processAudienceAndRag(session, userInput, sink)
            CampaignCreationStep.AUDIENCE_TARGETING -> processCreativeGeneration(session, userInput, sink)
            CampaignCreationStep.CREATIVE_GENERATION -> processMcpPublishing(session, userInput, sink)
            CampaignCreationStep.PUBLISHING_AND_PAYMENT -> handleAwaitingPayment(session, sink)
            CampaignCreationStep.COMPLETED -> handleCompleted(sink)
        }
    }

    /**
     * ШАГ 2: Анализ брифа через Qwen-35B + Поиск правил модерации в Qdrant
     */
    private fun processAudienceAndRag(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        sink.next("🧠 ИИ-Анализатор [Qwen-35B]: Изучаю ваш бриф и выделяю сегменты аудитории...\n")
        val analysisResult = aiInferenceService.analyzeBrief(userInput)
        val platforms = analysisResult.recommendations.map { it.platform }

        sink.next("📡 Qdrant [Размерность 2048]: Извлекаю правила модерации площадок...\n")
        val moderationRules = vectorStoreService.searchAdvertisingRules(platforms)

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.AUDIENCE_TARGETING,
                context = session.context.copy(
                    targetAudienceDescription = analysisResult.audienceDescription,
                    analysisResult = analysisResult,
                    moderationRules = moderationRules
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("📝 **Результаты анализа брифа:**\n")
        sink.next("• ЦА: ${analysisResult.audienceDescription}\n")
        sink.next("• Выбранные каналы: ${platforms.joinToString { it.name }}\n")
        sink.next("• База знаний Qdrant вернула локальные лимиты символов.\n\n")
        sink.next("🤖 **[Шаг 2/4]**: Переходим к генерации объявлений? Напишите **'Да'** или **'Генерируй'**.")
        sink.complete()
    }

    /**
     * ШАГ 3: Генерация текстов и картинок (Лимит токенов n_predict = 400)
     */
    private fun processCreativeGeneration(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        if (!isUserApproved(userInput)) {
            sink.next("⚠️ Жду подтверждения. Напишите **'Да'** или **'Генерируй'**, чтобы запустить написание текстов моделью Qwen-35B.")
            sink.complete()
            return
        }

        sink.next("⚡ Запускаю Qwen-35B (n_predict=400 лимит). Генерирую форматированные креативы...\n")
        val analysis = session.context.analysisResult ?: throw IllegalStateException("Данные анализа брифа отсутствуют в базе")
        val rules = session.context.moderationRules ?: ""

        val creatives = creativeGeneratorProcessor.generate(analysis, rules)

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.CREATIVE_GENERATION,
                context = session.context.copy(generatedCreatives = creatives),
                updatedAt = Instant.now()
            )
        )

        sink.next("✨ **Рекламные объявления сформированы:**\n")
        creatives.forEach { (platform, creative) ->
            sink.next("\n📢 **${platform.name}**\n🔹 Заголовок: ${creative.title}\n🔸 Текст: ${creative.bodyText}\n")
        }
        sink.next("\n🤖 **[Шаг 3/4]**: Публикуем эти креативы в кабинеты через Stateless MCP-сервер (8090)? Напишите **'Да'** или **'Публикуй'**.")
        sink.complete()
    }

    /**
     * ШАГ 4: Вызов MCP-инструментов на виртуальных потоках Loom
     */
    private fun processMcpPublishing(session: CampaignSession, userInput: String, sink: FluxSink<String>) {
        if (!isUserApproved(userInput)) {
            sink.next("⚠️ Ожидаю команду на отправку. Напишите **'Да'** или **'Публикуй'**, чтобы запустить RPC-транспорт.")
            sink.complete()
            return
        }

        sink.next("🚀 АГЕНТ: Вызываю инструменты автоматизации на сервере 8090 (Loom Virtual Threads)...\n")
        val creatives = session.context.generatedCreatives ?: throw IllegalStateException("Креативы не найдены")
        val analysis = session.context.analysisResult ?: throw IllegalStateException("Анализ отсутствует")
        val platforms = analysis.recommendations.map { it.platform }

        val publishResult = mcpPublishingService.publish(session.campaignId, platforms, creatives)
        publishResult.logs.forEach { mcpLog -> sink.next("$mcpLog\n") }

        val invoiceId = "INV-${System.currentTimeMillis()}-${session.campaignId}"

        sessionRepository.save(
            CampaignSession(
                campaignId = session.campaignId,
                currentStep = CampaignCreationStep.PUBLISHING_AND_PAYMENT,
                context = session.context.copy(
                    executionLogs = session.context.executionLogs + publishResult.logs,
                    platformCampaignIds = publishResult.campaignIds,
                    paymentInvoiceId = invoiceId
                ),
                updatedAt = Instant.now()
            )
        )

        sink.next("\n✅ **[Шаг 4/4]**: Кампании успешно выгружены!\n")
        sink.next("💳 Сформирован инвойс для перехода к оплате: **$invoiceId**\n")
        sink.next("🔗 Перенаправляю вас на биллинг-платформу...")
        sink.complete()
    }

    private fun handleAwaitingPayment(session: CampaignSession, sink: FluxSink<String>) {
        sink.next("🏁 Реклама уже опубликована. Ожидайте подтверждения транзакции по инвойсу ${session.context.paymentInvoiceId}.")
        sink.complete()
    }

    private fun handleCompleted(sink: FluxSink<String>) {
        sink.next("Заказ полностью завершен.")
        sink.complete()
    }

    private fun isUserApproved(input: String): Boolean {
        val clean = input.lowercase().trim()
        return clean in listOf("да", "ок", "публикуй", "подтверждаю", "yes", "ok", "генерируй")
    }
}