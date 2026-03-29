package com.micronaut.bug.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

import static com.micronaut.bug.config.ObservationConfig.TARGET_ID;

@Slf4j
@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 1)
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {

        String authHeader = request.getHeader(HttpHeaders.AUTHORIZATION);
        User user = null;

        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            String token = authHeader.substring(7);
            user = parseToken(token);
        }

        try {
            if (user != null) {
                // Привязываем пользователя к области видимости текущего вызова
                // Все, что внутри .run(), будет видеть этого юзера через SecurityContext.CURRENT_USER.get()
                MDC.put(TARGET_ID, user.id());
                final User finalUser = user;
                ScopedValue.where(SecurityContext.CURRENT_USER, finalUser)
                    .run(() -> {
                        try {
                            filterChain.doFilter(request, response);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
            } else {
                // Если пользователя нет, просто идем дальше по цепочке
                filterChain.doFilter(request, response);
            }
        } finally {
            MDC.remove(TARGET_ID);
        }
    }

    private User parseToken(String token) {
        // Ваша логика парсинга JWT
        return new User("123", "ivan_ivanov", "ADMIN");
    }
}