package com.altro.myorchestrator.config

import jakarta.servlet.Filter
import jakarta.servlet.FilterChain
import jakarta.servlet.ServletRequest
import jakarta.servlet.ServletResponse
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.stereotype.Component

@Component
@Order(Ordered.HIGHEST_PRECEDENCE) // Ставим самым первым в системе
class CorsPreflightFilter : Filter {

    override fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
        val req = request as HttpServletRequest
        val res = response as HttpServletResponse

        // Явно прописываем CORS-заголовки для ответа
        res.setHeader("Access-Control-Allow-Origin", "http://localhost:5173")
        res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE, PUT")
        res.setHeader("Access-Control-Allow-Headers", "*")
        res.setHeader("Access-Control-Allow-Credentials", "true")

        // Если это preflight-запрос OPTIONS, сразу возвращаем 200 OK и прерываем цепочку
        if ("OPTIONS" == req.method.uppercase()) {
            res.status = HttpServletResponse.SC_OK
            return
        }

        // Все остальные запросы (боевой POST) пропускаем дальше
        chain.doFilter(request, response)
    }
}
