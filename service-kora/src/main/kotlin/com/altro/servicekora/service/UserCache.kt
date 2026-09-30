package com.altro.servicekora.service

import com.altro.servicekora.model.UserEntity
import io.koraframework.cache.Cache

interface UserCache : Cache<Long, UserEntity>
