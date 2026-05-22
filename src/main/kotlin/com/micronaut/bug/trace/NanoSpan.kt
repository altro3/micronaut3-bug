package com.micronaut.bug.trace

import java.util.concurrent.atomic.AtomicIntegerFieldUpdater

class NanoSpan(
    @JvmField
    val traceId: String,
    @JvmField
    val spanId: String,
    @JvmField
    val parentId: String?,
    @JvmField
    val name: String,
    @JvmField
    val startEpochNanos: Long,
    @JvmField
    val sampled: Boolean,
    @JvmField
    val baggage: Map<String, String>?,
    @JvmField
    val propagationHeaders: Map<String, String>?,
    @JvmField
    val traceState: String?,
    @JvmField
    var parentSpan: NanoSpan? = null,
) {
    @Volatile
    var error: Throwable? = null

    @Suppress("unused")
    @Volatile
    private var active: Int = 1

    fun tryClose(): Boolean =
        ACTIVE_UPDATER.compareAndSet(this, 1, 0)

    companion object {
        private val ACTIVE_UPDATER = AtomicIntegerFieldUpdater.newUpdater(NanoSpan::class.java, "active")
    }
}
