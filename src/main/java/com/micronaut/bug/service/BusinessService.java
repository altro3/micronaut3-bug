package com.micronaut.bug.service;

import com.micronaut.bug.config.SecurityContext;
import com.micronaut.bug.config.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class BusinessService {

    public User processOrder() {
        // Проверяем, привязан ли пользователь
        User user = SecurityContext.getUser();
        if (user != null) {
            log.info("Обработка заказа для пользователя: {}", user.username());
        } else {
            log.warn("Анонимный запрос");
        }
        return user;
    }
}