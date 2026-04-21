package com.micronaut.bug.config

import io.github.oshai.kotlinlogging.KotlinLogging
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.MDC
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.http.HttpHeaders
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 1)
class JwtAuthenticationFilter : OncePerRequestFilter() {

    private val log = KotlinLogging.logger {}

    override fun doFilterInternal(
        rq: HttpServletRequest,
        rs: HttpServletResponse,
        chain: FilterChain
    ) {
        val authHeader = rq.getHeader(HttpHeaders.AUTHORIZATION)
        val user = parseToken("")
//      val user = if (authHeader != null && authHeader.startsWith(BEARER_PREFIX)) {
//            val token = authHeader.substring(BEARER_PREFIX.length)
//            parseToken(token)
//        } else {
//            null
//        }

        try {
            if (user != null) {
                MDC.put(TARGET_ID, user.id)
                SecurityContext.set(user)
            }
            chain.doFilter(rq, rs)
        } finally {
            MDC.remove(TARGET_ID)
            SecurityContext.clear()
        }
    }

    private fun parseToken(token: String): User {
        // Ваша логика парсинга JWT
        return User("123", "ivan_ivanov", "ADMIN")
    }

    companion object {
        private const val BEARER_PREFIX = "Bearer "
        private const val TARGET_ID = "targetId" // В оригинале было ObservationConfig.TARGET_ID
    }
}
