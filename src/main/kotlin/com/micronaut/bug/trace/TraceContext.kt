package com.micronaut.bug.trace

/**
 * Контейнер контекста распределенной трассировки.
 *
 * Объект является неизменяемым (immutable) и переносится между потоками исполнения
 * с помощью [TraceElement]. При создании дочерних спанов через [NanoTracer.startSpan]
 * данные копируются, обеспечивая наследование контекста по всей глубине вызовов.
 *
 * @property traceId Уникальный 32-символьный шестнадцатеричный идентификатор всей цепочки (trace).
 * @property spanId Уникальный 16-символьный шестнадцатеричный идентификатор текущего сегмента (span).
 * @property parentId Идентификатор родительского спана. Null для корневого спана (точки входа).
 * @property name Человекочитаемое имя операции (например, "GET /api/orders").
 * @property startEpochNanos Время начала операции в наносекундах (Unix Epoch).
 *
 * @property baggage Стандартный набор пар ключ-значение согласно спецификации W3C Baggage.
 * Передается в едином HTTP-заголовке "baggage". Предназначен для межсервисного
 * взаимодействия с системами, поддерживающими стандарт OpenTelemetry.
 *
 * @property propagationHeaders Набор пользовательских заголовков, которые должны пробрасываться
 * в неизменном виде ("как есть") во все исходящие HTTP-запросы. Ключи хранятся в нижнем
 * регистре для обеспечения регистронезависимого сравнения.
 *
 * @property traceState Вендор-специфичные данные согласно W3C Trace Context.
 * Хранятся как непрозрачная (opaque) строка и передаются без изменений для обеспечения
 * совместимости между различными вендорами трассировки.
 *
 * @property sampled Флаг сэмплирования, полученный от родительской системы (например, Istio)
 * или вычисленный на входе. Если false — спаны этой ветки не будут экспортироваться в Tempo,
 * чтобы снизить нагрузку на хранилище, за исключением случаев ошибок или медленных запросов.
 *
 */
data class TraceContext(
    val traceId: String,
    val spanId: String,
    val parentId: String? = null,
    val name: String,
    val startEpochNanos: Long,
    val baggage: Map<String, String>? = null,
    val propagationHeaders: Map<String, String>? = null,
    val traceState: String? = null,
    @Volatile
    var error: Throwable? = null,
    val sampled: Boolean = true,
)
