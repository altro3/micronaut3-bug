package com.micronaut.bug.trace

import com.micronaut.bug.trace.TraceUtil.ATTR_ERROR_MESSAGE
import io.opentelemetry.proto.trace.v1.Status.StatusCode

class TraceReport {
    var status: StatusCode = StatusCode.STATUS_CODE_UNSET
    var attrs: MutableMap<String, Any>? = null
        private set

    fun putAttr(key: String, value: Any) {
        if (attrs == null) attrs = HashMap(8)
        attrs!![key] = value
    }

    fun error(message: String?) {
        status = StatusCode.STATUS_CODE_ERROR
        message?.let { putAttr(ATTR_ERROR_MESSAGE, it) }
    }

    fun clear() {
        status = StatusCode.STATUS_CODE_UNSET
        attrs?.clear()
    }
}
