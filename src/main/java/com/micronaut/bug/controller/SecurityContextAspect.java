package com.micronaut.bug.controller;

import com.micronaut.bug.config.User;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.stereotype.Component;

import java.lang.annotation.Annotation;

//@Aspect
//@Component
public class SecurityContextAspect {
/*
    @Around("execution(* com.micronaut.bug.controller..*(.., @CurrentUser (*), ..))")
    public Object injectUser(ProceedingJoinPoint joinPoint) {
        return Mono.deferContextual(ctx -> {
            User user = ctx.getOrDefault("current-user", null);

            // Получаем все аргументы метода
            Object[] args = joinPoint.getArgs();
            var signature = (MethodSignature) joinPoint.getSignature();
            Annotation[][] parameterAnnotations = signature.getMethod().getParameterAnnotations();

            // Ищем, какой аргумент помечен @CurrentUser
            for (int i = 0; i < parameterAnnotations.length; i++) {
                var argFound = false;
                for (Annotation annotation : parameterAnnotations[i]) {
                    if (annotation instanceof CurrentUser) {
                        args[i] = user; // Подставляем юзера из контекста
                        argFound = true;
                        break;
                    }
                }
                if (argFound) {
                    break;
                }
            }

            try {
                // Выполняем метод контроллера с подставленным юзером
                Object result = joinPoint.proceed(args);

                // Если метод вернул Mono (а он должен), возвращаем его
                if (result instanceof Mono<?> mono) {
                    return mono;
                }
                return Mono.justOrEmpty(result);
            } catch (Throwable e) {
                return Mono.error(e);
            }
        });
    }*/
}