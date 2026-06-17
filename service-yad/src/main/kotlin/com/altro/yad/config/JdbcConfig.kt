package com.altro.yad.config

import com.altro.yad.model.Campaign.CampaignData
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
        CampaignDataWritingConverter(jsonMapper),
        CampaignDataReadingConverter(jsonMapper)
    )

    @WritingConverter
    class CampaignDataWritingConverter(
        private val jsonMapper: JsonMapper,
    ) : Converter<CampaignData, PGobject> {
        override fun convert(source: CampaignData): PGobject = PGobject().apply {
            type = "jsonb"
            value = jsonMapper.writeValueAsString(source)
        }
    }

    @ReadingConverter
    class CampaignDataReadingConverter(
        private val jsonMapper: JsonMapper,
    ) : Converter<PGobject, CampaignData> {
        override fun convert(source: PGobject): CampaignData =
            jsonMapper.readValue(source.value, CampaignData::class.java)
    }
}
