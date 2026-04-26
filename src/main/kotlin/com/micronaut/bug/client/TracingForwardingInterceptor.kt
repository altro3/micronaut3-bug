package com.micronaut.bug.client

import com.micronaut.bug.config.ServerLoggingFilter.Companion.MDC_RQ_ID
import org.slf4j.MDC
import org.springframework.http.HttpRequest
import org.springframework.http.client.ClientHttpRequestExecution
import org.springframework.http.client.ClientHttpRequestInterceptor
import org.springframework.http.client.ClientHttpResponse

/**
 * Интерцептор для обеспечения сквозной трассировки (Distributed Tracing).
 *
 * Основная задача: извлечь уникальный идентификатор запроса (x-req-id) из текущего
 * контекста логирования (MDC) и передать его в заголовках исходящего HTTP-запроса.
 * Это позволяет связать логи вызывающего сервиса и вызываемого API в единую цепочку.
 */
class TracingForwardingInterceptor : ClientHttpRequestInterceptor {

    /**
     * Перехватывает исходящий запрос для добавления заголовков трассировки.
     *
     * @param request объект исходящего запроса
     * @param body тело запроса в виде массива байтов
     * @param execution объект для продолжения цепочки выполнения запроса
     * @return ответ от внешнего сервиса
     */
    override fun intercept(
        request: HttpRequest,
        body: ByteArray,
        execution: ClientHttpRequestExecution
    ): ClientHttpResponse {
        // Пытаемся получить Trace ID, который был сгенерирован или получен серверным фильтром
        val traceId = MDC.get(MDC_RQ_ID)

        // Если идентификатор найден, обогащаем им заголовки внешнего вызова
        if (!traceId.isNullOrBlank()) {
            request.headers.set(HEADER_X_RQ_ID, traceId)
        }

        return execution.execute(request, body)
    }

    companion object {
        /**
         * Стандартный заголовок для передачи идентификатора запроса между микросервисами
         */
        private const val HEADER_X_RQ_ID = "x-rq-id"
    }
}
