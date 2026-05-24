package com.altro.common.log

import com.altro.common.log.ContentTypeAnalyzer.isJsonContent
import com.altro.common.log.ContentTypeAnalyzer.isMultipart
import jakarta.servlet.ReadListener
import jakarta.servlet.ServletInputStream
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletRequestWrapper
import jakarta.servlet.http.Part
import org.springframework.http.MediaType
import org.springframework.util.unit.DataSize
import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.InputStreamReader

object RequestWrapperFactory {

    fun wrap(rq: HttpServletRequest, maxPayloadSize: DataSize): HttpServletRequest {
        return when {
            isMultipart(rq.contentType) -> MultipartTypeWrapper(rq)
            rq.contentLengthLong > maxPayloadSize.toBytes() -> rq
            else -> CachedBodyRequestWrapper(rq, rq.inputStream.readAllBytes())
        }
    }

    class CachedBodyRequestWrapper(
        rq: HttpServletRequest,
        val body: ByteArray,
    ) : HttpServletRequestWrapper(rq) {
        override fun getInputStream(): ServletInputStream {
            val stream = ByteArrayInputStream(body)
            return object : ServletInputStream() {
                override fun read(): Int = stream.read()
                override fun isFinished(): Boolean = stream.available() == 0
                override fun isReady(): Boolean = true
                override fun setReadListener(readListener: ReadListener?) {}
            }
        }

        override fun getReader(): BufferedReader = BufferedReader(InputStreamReader(getInputStream()))
    }

    class MultipartTypeWrapper(
        rq: HttpServletRequest,
    ) : HttpServletRequestWrapper(rq) {
        private val cachedParts by lazy {
            super.getParts().map { part ->
                val bytes = part.inputStream.use { it.readBytes() }
                val ct = part.contentType
                val isJson = (ct == null || ct == MediaType.APPLICATION_OCTET_STREAM_VALUE) && isJsonContent(bytes)
                val finalCt = if (isJson) MediaType.APPLICATION_JSON_VALUE else ct
                ObservedPart(part, finalCt, bytes)
            }
        }

        override fun getParts(): Collection<Part> = cachedParts
        override fun getPart(name: String): Part? = cachedParts.find { it.name == name }
    }

    class ObservedPart(
        private val original: Part,
        private val overriddenContentType: String?,
        private val bytes: ByteArray,
    ) : Part by original {
        override fun getContentType() = overriddenContentType
        override fun getInputStream() = bytes.inputStream()
        override fun getSize() = bytes.size.toLong()
        fun getContentBytes() = bytes
    }
}
