package com.altro.servicekora.service

import com.altro.servicekora.model.UserEntity
import io.koraframework.cache.annotation.Cache
import io.koraframework.cache.caffeine.CaffeineCache

@Cache("cache.users")
interface UserCache : CaffeineCache<Long, UserEntity>
