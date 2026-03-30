package com.micronaut.bug.service;

import com.micronaut.bug.config.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class BusinessService {

    public User processOrder(User user) {
        if (user == null) {
            log.warn("Пользователь не найден, пропускаем заказ");
            return null;
        }
        log.info("Оформляем заказ для пользователя: {} (ID: {})", user.username(), user.id());
        return user;
    }
}