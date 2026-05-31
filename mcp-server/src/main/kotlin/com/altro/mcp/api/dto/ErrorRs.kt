package com.altro.mcp.api.dto

/**
 * Стандартный формат ответа об ошибке в системе.
 *
 * @property rqId Сквозной идентификатор запроса (TraceId).
 * @property code Строковый код ошибки для фронтенда или клиента.
 */
data class ErrorRs(
    val rqId: String?,
    val code: String,
    val message: String? = null
)
