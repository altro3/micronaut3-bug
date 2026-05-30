package com.altro.common.log

object LogConst {
    // Маркеры состояния тела
    const val BODY_EMPTY = "[EMPTY]"
    const val BODY_BINARY = "[BINARY DATA]"
    const val BODY_TOO_LARGE = "[BODY TOO LARGE TO LOG]"
    const val BODY_LOG_DISABLED = "[BODY LOGGING DISABLED]"
    const val BODY_MULTIPART_RS = "[MULTIPART RAW DISABLED]"
    const val BODY_STREAM = "[STREAMING CONTENT LOGGING DISABLED]"
    
    // Статусы ответов
    const val STATUS_UNKNOWN = "UNKNOWN"
    const val STATUS_UNDEFINED = "UNDEFINED"

    // Метки и префиксы для Multipart
    const val PART_UNKNOWN = "UNKNOWN"
    const val TEMPLATE_PART_PREFIX = "  [PART] -> Name: %s%s | %s"
    const val TEMPLATE_PART_FILE_INFO = "%s (File: %s, Size: %d bytes)"
    
    // Служебные атрибуты запросов (Spring / Servlet)
    const val ATTR_SKIP_LOGGING = "alg.skipLogging"
    const val ATTR_EXT_RQ_ID = "alg.extRqId"
    
    // Ключи MDC
    const val MDC_TYPE = "type"
    const val MDC_TARGET = "target"

    // Разделители и протоколы
    const val ENCODING_GZIP = "gzip"
    const val MARKER_HTTP_PROTOCOL = "http"
    const val SLASH = "/"
    const val STRING_EMPTY = ""
}
