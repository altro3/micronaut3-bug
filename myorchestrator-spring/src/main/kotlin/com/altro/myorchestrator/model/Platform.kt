package com.altro.myorchestrator.model

import com.fasterxml.jackson.annotation.JsonAlias

enum class Platform {
    @JsonAlias("YANDEX_DIRECT", "YANDEX__DIRECT", "Yandex", "yandex_direct")
    YANDEX_DIRECT,
    @JsonAlias("VK_ADS", "VK__ADS", "vk_ads")
    VK_ADS,
}