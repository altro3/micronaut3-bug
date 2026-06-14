package com.altro.myorchestrator.api.dto.chat

data class ChatRs(
    val id: String,
    val `object`: String,
    val created: Long,
    val choices: List<ChatChoice>
) {

    data class ChatChoice(
        val index: Int,
        val message: ChatMessageDto,
        val finish_reason: String
    )
}