package com.altro.service2.api

import com.altro.common.trace.NanoTracer
import com.altro.common.util.api.DefExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice

@RestControllerAdvice
class ExceptionHandler(
    tracer: NanoTracer? = null
) : DefExceptionHandler(tracer)
