package com.altro.common.util.api.config

import org.springframework.http.converter.HttpMessageConverters.ServerBuilder
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer
import tools.jackson.databind.json.JsonMapper

open class DefWebConfig(
    private val jsonMapper: JsonMapper,
) : WebMvcConfigurer {

    override fun configureMessageConverters(builder: ServerBuilder) {
        builder.withJsonConverter(JacksonJsonHttpMessageConverter(jsonMapper))
    }
}