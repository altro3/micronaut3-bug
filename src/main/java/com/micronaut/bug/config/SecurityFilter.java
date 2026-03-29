package com.micronaut.bug.config;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

import static com.micronaut.bug.config.ObservationConfig.TARGET_ID;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 1)
public class SecurityFilter implements WebFilter {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        String authHeader = exchange.getRequest().getHeaders().getFirst(HttpHeaders.AUTHORIZATION);

        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return chain.filter(exchange);
        }

        // Имитация парсинга JWT
        String token = authHeader.substring(7);
        User user = parseToken(token);

        return chain.filter(exchange)
            // Кладем в реактивный контекст.
            // Отсюда Context Propagation заберет его и положит в ScopedValue
            .contextWrite(ctx -> ctx.put(TARGET_ID, user.id())
                .put("current-user", user));
    }

    private User parseToken(String token) {
        // Здесь ваша логика (nimbus-jose-jwt или jjwt)
        return new User("123", "ivan_ivanov", "ADMIN");
    }
}
