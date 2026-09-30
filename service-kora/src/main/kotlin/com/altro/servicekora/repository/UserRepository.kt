package com.altro.servicekora.repository

import com.altro.servicekora.model.UserEntity
import io.koraframework.database.common.annotation.Query
import io.koraframework.database.common.annotation.Repository
import io.koraframework.database.jdbc.JdbcRepository

@Repository
interface UserRepository : JdbcRepository {

    @Query("SELECT id, name, email FROM users WHERE id = :id")
    fun findById(id: Long): UserEntity?

    @Query("INSERT INTO users(name, email) VALUES (:name, :email) RETURNING id, name, email")
    fun save(name: String, email: String): UserEntity
}
