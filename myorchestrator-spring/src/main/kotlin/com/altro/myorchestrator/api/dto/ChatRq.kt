package com.altro.myorchestrator.api.dto

import jakarta.validation.constraints.NotEmpty
import java.util.UUID

data class ChatRq(
    val sessionId: UUID,
    @field:NotEmpty
    val messages: List<ChatMessageDto>
) {

    data class ChatMessageDto(
        val role: String,
        val content: String,
    )
}