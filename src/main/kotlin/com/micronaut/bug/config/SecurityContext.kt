package com.micronaut.bug.config

object SecurityContext {

    private val HOLDER = ThreadLocal<User>()

    fun set(user: User) {
        HOLDER.set(user)
    }

    fun getUser(): User? {
        return HOLDER.get()
    }

    fun clear() {
        HOLDER.remove()
    }
}
