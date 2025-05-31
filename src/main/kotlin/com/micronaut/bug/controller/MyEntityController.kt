package com.micronaut.bug.controller

import com.fasterxml.jackson.annotation.*
import io.micronaut.http.annotation.*
import io.micronaut.http.client.BlockingHttpClient
import io.micronaut.http.client.HttpClient
import io.micronaut.http.client.annotation.Client
import io.micronaut.scheduling.TaskExecutors
import io.micronaut.scheduling.annotation.ExecuteOn
import io.micronaut.serde.annotation.Serdeable
import io.swagger.v3.oas.annotations.media.Schema
import jakarta.annotation.Nullable
import jakarta.validation.Valid
import jakarta.validation.constraints.NotNull

@ExecuteOn(TaskExecutors.BLOCKING)
@Controller
open class MyEntityController(
    @Client("local") httpClient: HttpClient
) {
    private val httpClient: BlockingHttpClient

    init {
        this.httpClient = httpClient.toBlocking()
    }

    @Post("/test{/apiVersion}")
    open fun test(
        @Header("X-Favor-Token") @Nullable xFavorToken: String? = null,
        @PathVariable("apiVersion") @Nullable apiVersion: MyEnum? = MyEnum.V5,
        @Header("Content-Type") @Nullable contentType: String? = "application/json"
    ): String? {
        return null
    }

    @Get("/test2/{param}")
    open fun test2(param: EnumParam?, @QueryValue qparam: EnumParam?): EnumParam? {
        return param
    }

    @Post("/test")
    open fun test(
        @QueryValue @Nullable apiVersion: String = "myVersion",
        @Body body: BodyDto,
    ): String? {
        return null
    }

    @Serdeable
    open class BodyDto(
        val value: String,
        @field:Nullable
        var currency: TestCurrency = TestCurrency.USD,
    )

    @Serdeable
    enum class TestCurrency {
        USD,
        RUB,
    }

    @Serdeable
    @JsonPropertyOrder(
        Test.JSON_PROPERTY_ID,
        Test.JSON_PROPERTY_CURRENCY,
    )
    open class Test(

        /**
         * TEST
         */
        @field:Nullable
        @field:Schema(name = "id", description = "TEST", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        @field:JsonProperty(JSON_PROPERTY_ID)
        @field:JsonInclude(JsonInclude.Include.USE_DEFAULTS)
        var id: String? = null,

        @field:Schema(name = "currency", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        @field:JsonProperty(JSON_PROPERTY_CURRENCY)
        @field:JsonInclude(JsonInclude.Include.USE_DEFAULTS)
        var currency: TestCurrency = TestCurrency.USD,
    ) {

        companion object {

            const val JSON_PROPERTY_ID = "id"
            const val JSON_PROPERTY_CURRENCY = "currency"
        }
    }

    @Serdeable
    @JsonPropertyOrder(
        TESTRequest.JSON_PROPERTY_VALUE,
        TESTRequest.JSON_PROPERTY_CURRENCY,
    )
    open class TESTRequest(

        @field:Nullable
        @field:Schema(name = "value", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        @field:JsonProperty(JSON_PROPERTY_VALUE)
        @field:JsonInclude(JsonInclude.Include.USE_DEFAULTS)
        var `value`: String? = null,

        @field:Schema(name = "currency", requiredMode = Schema.RequiredMode.NOT_REQUIRED)
        @field:JsonProperty(JSON_PROPERTY_CURRENCY)
        @field:JsonInclude(JsonInclude.Include.USE_DEFAULTS)
        var currency: MyCur = MyCur.USD,
    ) {

        companion object {

            const val JSON_PROPERTY_VALUE = "value"
            const val JSON_PROPERTY_CURRENCY = "currency"
        }
    }

    @Serdeable
    enum class MyCur(
        @get:JsonValue val value: String,
    ) {

        @JsonProperty("USD")
        USD("USD"),

        @JsonProperty("SAR")
        SAR("SAR"),
        ;

        override fun toString(): String = value

        companion object {

            @JvmField
            val VALUE_MAPPING = entries.associateBy { it.value }

            /**
             * Create this enum from a value.
             *
             * @param value The value
             *
             * @return The enum
             */
            @JsonCreator
            @JvmStatic
            fun fromValue(value: String): MyCur {
                require(VALUE_MAPPING.containsKey(value)) { "Unexpected value '$value'" }
                return VALUE_MAPPING[value]!!
            }
        }
    }
}
