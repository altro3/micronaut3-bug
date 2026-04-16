package com.micronaut.bug.controller

import com.micronaut.bug.config.User
import org.aspectj.lang.ProceedingJoinPoint
import org.aspectj.lang.annotation.Around
import org.aspectj.lang.annotation.Aspect
import org.aspectj.lang.reflect.MethodSignature
import org.springframework.stereotype.Component
//import reactor.core.publisher.Mono

//@Aspect
//@Component
class SecurityContextAspect {
/*

    @Around("execution(* com.micronaut.bug.controller..*(.., @CurrentUser (*), ..))")
    fun injectUser(joinPoint: ProceedingJoinPoint): Any? {
        return Mono.deferContextual { ctx ->
            val user = ctx.getOrDefault<User>("current-user", null)

            // Получаем все аргументы метода
            val args = joinPoint.args
            val signature = joinPoint.signature as MethodSignature
            val parameterAnnotations = signature.method.parameterAnnotations

            // Ищем, какой аргумент помечен @CurrentUser
            for (i in parameterAnnotations.indices) {
                val annotations = parameterAnnotations[i]
                if (annotations.any { it is CurrentUser }) {
                    args[i] = user // Подставляем юзера из контекста
                    break
                }
            }

            try {
                // Выполняем метод контроллера с подставленным юзером
                val result = joinPoint.proceed(args)

                // Если метод вернул Mono, возвращаем его, иначе оборачиваем
                if (result is Mono<*>) {
                    result
                } else {
                    Mono.justOrEmpty(result)
                }
            } catch (e: Throwable) {
                Mono.error<Any>(e)
            }
        }
    }
*/
}
