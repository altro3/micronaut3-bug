package com.altro.myorchestrator.config

import com.altro.myorchestrator.model.CampaignSession.SessionContext
import org.postgresql.util.PGobject
import org.springframework.context.annotation.Configuration
import org.springframework.core.convert.converter.Converter
import org.springframework.data.convert.ReadingConverter
import org.springframework.data.convert.WritingConverter
import org.springframework.data.jdbc.repository.config.AbstractJdbcConfiguration
import tools.jackson.databind.json.JsonMapper

@Configuration
class JdbcConfig(
    private val jsonMapper: JsonMapper,
) : AbstractJdbcConfiguration() {

    override fun userConverters(): List<*> = listOf(
        SessionContextWritingConverter(jsonMapper),
        SessionContextReadingConverter(jsonMapper)
    )

    @WritingConverter
    class SessionContextWritingConverter(
        private val mapper: JsonMapper,
    ) : Converter<SessionContext, PGobject> {
        override fun convert(source: SessionContext): PGobject = PGobject().apply {
            type = "jsonb"
            value = mapper.writeValueAsString(source)
        }
    }

    @ReadingConverter
    class SessionContextReadingConverter(
        private val mapper: JsonMapper,
    ) : Converter<PGobject, SessionContext?> {
        override fun convert(source: PGobject): SessionContext? =
            source.value?.let { mapper.readValue(it, SessionContext::class.java) }
    }
}
