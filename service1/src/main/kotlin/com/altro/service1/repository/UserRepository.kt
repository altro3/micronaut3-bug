package com.altro.service1.repository

import com.altro.service1.model.UserEntity
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository

@Repository
interface UserRepository : CrudRepository<UserEntity, Long>
