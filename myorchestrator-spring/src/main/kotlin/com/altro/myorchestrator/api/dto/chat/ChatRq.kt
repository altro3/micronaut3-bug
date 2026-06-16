package com.altro.myorchestrator.api.dto.chat

import jakarta.validation.constraints.NotEmpty

data class ChatRq(
    @field:NotEmpty
    val messages: List<ChatMessageDto>
) {

    data class ChatMessageDto(
        val role: String,
        val content: String,
    )
}
