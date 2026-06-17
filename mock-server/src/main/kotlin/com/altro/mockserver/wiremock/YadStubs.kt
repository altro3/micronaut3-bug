package com.altro.mockserver.wiremock

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.containing
import com.github.tomakehurst.wiremock.client.WireMock.get
import com.github.tomakehurst.wiremock.client.WireMock.matchingJsonPath
import com.github.tomakehurst.wiremock.client.WireMock.notMatching
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import org.springframework.http.HttpHeaders.CONTENT_LENGTH
import org.springframework.http.HttpHeaders.CONTENT_TYPE
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType

object YadStubs {

    fun setupStubs(wireMockServer: WireMockServer) {

        val regionsRs = """
            {
              "result": {
                "GeoRegions": [
                  { "RegionId": 1, "RegionName": "Москва и Московская область" },
                  { "RegionId": 2, "RegionName": "Санкт-Петербург и Ленинградская область" },
                  { "RegionId": 3, "RegionName": "Новосибирская область (Сибирь)" },
                  { "RegionId": 4, "RegionName": "Бердск" }
                ]
              }
            }
        """.trimIndent()

        // 1. Заглушка для получения регионов: GET /yad/dict/regions
        wireMockServer.stubFor(
            get(urlEqualTo("/yad/dict/regions"))
                .willReturn(
                    aResponse()
                        .withStatus(HttpStatus.OK.value())
                        .withHeader(CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                        .withBody(regionsRs)
                )
        )

        val campaignCreateOkRs = """
            {
              "result": {
                "AddResults": [
                  {
                    "Id": 777123456,
                    "Errors": null
                  }
                ]
              }
            }
        """.trimIndent()
        // 2. УСПЕШНЫЙ СЦЕНАРИЙ: Создание кампании: POST /yad/campaigns/create (Обычный запрос)
        wireMockServer.stubFor(
            post(urlEqualTo("/yad/campaigns/create"))
                .withHeader(CONTENT_TYPE, containing("application/json"))
                // Если в JSON-теле нет триггера ошибки — возвращаем успешный ID
                .withRequestBody(notMatching(".*ERROR_TRIGGER.*"))
                .willReturn(
                    aResponse()
                        .withStatus(HttpStatus.OK.value())
                        .withHeader(CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                        .withBody(campaignCreateOkRs)
                )
        )

        val campaignCreateErrorRs = """
            {
              "result": {
                "AddResults": [
                  {
                    "Id": null,
                    "Errors": [
                      {
                        "Code": 4001,
                        "Message": "Inconsistent targeting parameters",
                        "Details": "Выбранный регион не соответствует настройкам аккаунта"
                      }
                    ]
                  }
                ]
              }
            }
        """.trimIndent()

        // 3. СЦЕНАРИЙ ОШИБКИ: Создание кампании: POST /yad/campaigns/create (Признак ошибки)
        wireMockServer.stubFor(
            post(urlEqualTo("/yad/campaigns/create"))
                .withHeader(CONTENT_TYPE, containing("application/json"))
                // Триггеримся на специальное слово внутри JSON-тела для теста статуса SYNC_ERROR
                .withRequestBody(matchingJsonPath("$.params.Campaigns[0].Name", containing("ERROR_TRIGGER")))
                .willReturn(
                    aResponse()
                        .withStatus(HttpStatus.OK.value()) // Сохраняем логику ответа 200 со списком ошибок в теле
                        .withHeader(CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                        .withBody(campaignCreateErrorRs)
                )
        )
    }
}
