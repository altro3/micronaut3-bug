package com.altro.servicekora.repository

interface UserRepository
/*
@Repository
interface UserRepository : JdbcRepository {

    @Query("SELECT id, name, email FROM users WHERE id = :id")
    fun findById(id: Long): UserEntity?

    @Query("INSERT INTO users(name, email) VALUES (:name, :email) RETURNING id, name, email")
    fun save(name: String, email: String): UserEntity

    @Query("SELECT 1")
    fun checkConnection(): Boolean
}
*/
