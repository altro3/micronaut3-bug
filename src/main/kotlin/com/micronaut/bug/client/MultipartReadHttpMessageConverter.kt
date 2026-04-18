package com.micronaut.bug.client

import org.springframework.http.HttpInputMessage
import org.springframework.http.HttpOutputMessage
import org.springframework.http.MediaType
import org.springframework.http.converter.AbstractHttpMessageConverter
import org.springframework.util.LinkedMultiValueMap
import org.springframework.util.MultiValueMap
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import java.io.ByteArrayOutputStream

/**
 * Профессиональная реализация конвертера для чтения multipart/form-data.
 * Преобразует входящий поток в MultiValueMap<String, HttpEntity<ByteArray>>.
 */
class MultipartReadHttpMessageConverter : AbstractHttpMessageConverter<MultiValueMap<String, *>>(MediaType.MULTIPART_FORM_DATA) {

    override fun supports(clazz: Class<*>): Boolean = MultiValueMap::class.java.isAssignableFrom(clazz)

    override fun writeInternal(t: MultiValueMap<String, *>, outputMessage: HttpOutputMessage) {
        throw UnsupportedOperationException("Use AllEncompassingFormHttpMessageConverter for writing")
    }

    override fun readInternal(
        clazz: Class<out MultiValueMap<String, *>>,
        inputMessage: HttpInputMessage
    ): MultiValueMap<String, Any> {
        val contentType = inputMessage.headers.contentType ?: throw IllegalArgumentException("No Content-Type")
        val boundary = contentType.parameters["boundary"]?.toByteArray()
            ?: throw IllegalArgumentException("No boundary in Content-Type")

        val result = LinkedMultiValueMap<String, Any>()
        val body = inputMessage.body.readAllBytes()

        val separator = "--".toByteArray() + boundary
        val endMarker = separator + "--".toByteArray()

        var pos = findIndex(body, separator)
        if (pos == -1) return result

        while (pos < body.size) {
            val nextSeparator = findIndex(body, separator, pos + separator.size)
            if (nextSeparator == -1) break

            val partBytes = body.copyOfRange(pos + separator.size, nextSeparator)
            if (partBytes.isNotEmpty()) {
                parsePart(partBytes, result)
            }

            if (isEndMarker(body, nextSeparator, endMarker)) break
            pos = nextSeparator
        }

        return result
    }

    private fun parsePart(bytes: ByteArray, result: MultiValueMap<String, Any>) {
        val doubleNewLine = "\r\n\r\n".toByteArray()
        val headerEnd = findIndex(bytes, doubleNewLine)
        if (headerEnd == -1) return

        val headerBytes = bytes.copyOfRange(0, headerEnd)
        val contentBytes = bytes.copyOfRange(headerEnd + doubleNewLine.size, bytes.size).trimCrlf()

        val headers = HttpHeaders()
        String(headerBytes).lines().filter { it.isNotBlank() }.forEach { line ->
            val (key, value) = line.split(":", limit = 2).map { it.trim() }
            headers.add(key, value)
        }

        val disposition = headers.getFirst(HttpHeaders.CONTENT_DISPOSITION) ?: return
        val name = disposition.substringAfter("name=\"").substringBefore("\"")

        // Оборачиваем в HttpEntity, чтобы сохранить заголовки каждой части (Content-Type и т.д.)
        result.add(name, HttpEntity(contentBytes, headers))
    }

    private fun findIndex(source: ByteArray, target: ByteArray, start: Int = 0): Int {
        for (i in start..source.size - target.size) {
            if (target.indices.all { j -> source[i + j] == target[j] }) return i
        }
        return -1
    }

    private fun isEndMarker(body: ByteArray, pos: Int, marker: ByteArray): Boolean {
        return pos + marker.size <= body.size &&
                marker.indices.all { i -> body[pos + i] == marker[i] }
    }

    private fun ByteArray.trimCrlf(): ByteArray {
        return if (this.size >= 2 && this[size - 2] == '\r'.code.toByte() && this[size - 1] == '\n'.code.toByte()) {
            this.copyOfRange(0, size - 2)
        } else this
    }
}
