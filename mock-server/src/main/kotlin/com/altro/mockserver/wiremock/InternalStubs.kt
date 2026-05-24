package com.altro.mockserver.wiremock

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.absent
import com.github.tomakehurst.wiremock.client.WireMock.containing
import com.github.tomakehurst.wiremock.client.WireMock.get
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.client.WireMock.urlMatching
import java.io.ByteArrayOutputStream
import java.util.zip.GZIPOutputStream

object InternalStubs {

    fun setupStubs(wireMockServer: WireMockServer) {
        // 1. Тест Ping — проверяем логирование пустого тела и статус кода
        wireMockServer.stubFor(
            get(urlEqualTo("/testtracelog/health/check"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "text/plain")
                        .withBody("OK")
                )
        )

        // 2. Тест JSON — проверяем Pretty Print в твоем интерцепторе
        wireMockServer.stubFor(
            post(urlEqualTo("/testtracelog/v1/data/sync"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        // Специально пишем в одну строку, чтобы проверить, что твой интерцептор развернет это в Pretty Print
                        .withBody("""{"status":"SUCCESS","data":{"id":123,"integration":"verified","timestamp":"2026-04-18T12:00:00"}}""")
                )
        )

        // 3. Тест Process File — проверяем детектирование бинарных данных (BINARY DATA в логах)
        wireMockServer.stubFor(
            post(urlEqualTo("/testtracelog/v1/files/upload"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/octet-stream")
                        .withHeader("Content-Disposition", "attachment; filename=\"response-file.dat\"")
                        .withBody(byteArrayOf(1, 2, 3, 4, 5, 10, 13))
                )
        )

        // 4. Тест Complex Multipart — возвращаем честный Multipart ответ
        val boundary = "vX2930293jksldj"

        // Собираем тело из байтов, чтобы бинарный кусок прошел корректно
        val multipartBody = ByteArrayOutputStream().apply {
            // Часть 1: JSON
            write("--$boundary\r\n".toByteArray())
            write("Content-Disposition: form-data; name=\"status\"\r\n".toByteArray())
            write("Content-Type: application/json\r\n\r\n".toByteArray())
            write("{\"result\": \"all_parts_received\"}\r\n".toByteArray())

            // Часть 2: Текстовый файл
            write("--$boundary\r\n".toByteArray())
            write("Content-Disposition: form-data; name=\"report\"; filename=\"summary.txt\"\r\n".toByteArray())
            write("Content-Type: text/plain\r\n\r\n".toByteArray())
            write("Everything went perfect!\r\n".toByteArray())

            // Часть 3: Бинарный файл (с NULL-байтом для проверки твоего логера)
            write("--$boundary\r\n".toByteArray())
            write("Content-Disposition: form-data; name=\"image\"; filename=\"result.png\"\r\n".toByteArray())
            write("Content-Type: image/png\r\n\r\n".toByteArray())
            write(byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 1, 2, 3)) // Сигнатура PNG + NULL
            write("\r\n".toByteArray())

            // Финальная граница
            write("--$boundary--\r\n".toByteArray())
        }.toByteArray()

        wireMockServer.stubFor(
            post(urlMatching("/testtracelog/v2/complex-process/.*"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "multipart/form-data; boundary=$boundary")
                        .withBody(multipartBody)
                )
        )

        // 5. Тест гигантского JSON — проверка умной обрезки (начало...конец)
        val bigData = StringBuilder().apply {
            append("""{"status": "START_OF_BIG_JSON", "items": [""")
            // Генерируем "мусор" на ~32KB
            repeat(1000) { i ->
                append("""{"id": $i, "value": "some repetitive long string data to fill space"},""")
            }
            append("""{"id": 9999, "value": "END_OF_BIG_JSON"}]}""")
        }.toString()

        wireMockServer.stubFor(
            get(urlEqualTo("/testtracelog/v1/data/huge"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "text/plain")
                        .withBody(bigData)
                )
        )

        wireMockServer.stubFor(
            post(urlEqualTo("/testtracelog/v1/data/gzip"))
                // Используем встроенный матчер WireMock для заголовка
                .withHeader("Content-Encoding", containing("gzip"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withHeader("Content-Encoding", "gzip")
                        .withBody(convertToGzip("""{"status":"gzip_success","message":"Data processed"}"""))
                )
        )

        wireMockServer.stubFor(
            post(urlEqualTo("/testtracelog/v1/data/manual-gzip"))
                // Используем встроенный матчер WireMock для заголовка
                .withHeader("Content-Encoding", containing("gzip"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "octet/stream")
                        .withHeader("Content-Encoding", "gzip")
                        .withBody(convertToGzip("""{"status":"gzip_success","message":"Data processed"}"""))
                )
        )

        // В класс ExternalMockServer, метод setupStubs()
        wireMockServer.stubFor(
            post(urlEqualTo("/testtracelog/v1/data/plain-to-gzip"))
                // Проверяем, что клиент НЕ прислал сжатые данные
                .withHeader("Content-Encoding", absent())
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withHeader("Content-Encoding", "gzip") // Сервер сжимает ответ
                        .withBody(convertToGzip("""{"status":"success","info":"received plain, sent gzip"}"""))
                )
        )

        val hugeSize = 5 * 1024 * 1024 // 5MB
        val hugeByteArray = ByteArray(hugeSize) { 0x41.toByte() } // Заполним буквой 'A'

        wireMockServer.stubFor(
            post(urlEqualTo("/v1/data/huge-proxy"))
                .willReturn(
                    aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "text/plain")
                        // WireMock выставит Content-Length: 5242880
                        .withBody(hugeByteArray)
                )
        )
    }

    private fun convertToGzip(content: String): ByteArray {
        val baos = ByteArrayOutputStream()
        GZIPOutputStream(baos).use { it.write(content.toByteArray()) }
        return baos.toByteArray()
    }
}