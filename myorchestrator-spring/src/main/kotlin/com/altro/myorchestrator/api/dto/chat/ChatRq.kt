package com.altro.myorchestrator.api.dto.chat

data class ChatRq(
    val model: String,
    val messages: List<ChatMessageDto>
)
